import torch
from torch import Tensor, nn
import math
import torch.nn.functional as F
from typing import Tuple, Type
from .feat_pe import get_feat_pe
from .meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator,TokenMetric

class MLPBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        act: Type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = act()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin2(self.act(self.lin1(x)))

class TwoWayAttentionBlock(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = 2048,
        activation: Type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = 2,
        skip_first_layer_pe: bool = False,
    ) -> None:
        """
        A transformer block with four layers: (1) self-attention of sparse
        inputs, (2) cross attention of sparse inputs to dense inputs, (3) mlp
        block on sparse inputs, and (4) cross attention of dense inputs to sparse
        inputs.

        Arguments:
          embedding_dim (int): the channel dimension of the embeddings
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()

        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )

        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
        self, queries: Tensor, keys: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        

        # Cross attention block, tokens attending to image embedding
        q = queries
        if key_pe is not None:
            k = keys + key_pe
        else:
            k = keys
        attn_out, attn_score = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # MLP block
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # Cross attention block, image embedding attending to tokens
        q = queries
        if key_pe is not None:
            k = keys + key_pe
        else:
            k = keys
        attn_out,_ = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)
        
        return queries, keys, attn_score

class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."

        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        b, n, c = x.shape
        x = x.reshape(b, n, num_heads, c // num_heads)
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn_ = attn / math.sqrt(c_per_head)
        attn = torch.softmax(attn_, dim=-1)  # (bs, num_heads, num_cls, L)

        # Get output
        out = attn @ v
        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out, attn_


class WSCerPartial(MetaClassifier):
    def __init__(self, args):
        input_embed_dim = args.neck_output_dim[0]
        num_classes = args.num_classes
        evaluator = build_evaluator([TokenMetric()])
        super(WSCerPartial, self).__init__(evaluator, **args)

        depth = 2
        num_heads = 8
        mlp_dim = 2048
        self.pos_add_type = 'sam' # 'sam','query2label',None
        self.num_classes = num_classes

        self.conv_fc = nn.Linear(input_embed_dim, 1)
        self.cls_tokens = nn.Embedding(num_classes, input_embed_dim)
        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=input_embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=nn.ReLU,
                    attention_downsample_rate=2,
                    skip_first_layer_pe=(i == 0),
                )
            )

    def calc_logits(self, img_tokens: torch.Tensor):
        bs, num_tokens, embed_dim = img_tokens.shape
        feat_size = int(math.sqrt(num_tokens))
        queries = self.cls_tokens.weight.unsqueeze(0).expand(bs, -1, -1)
        key_pe = None
        if self.pos_add_type is not None:
            # key_pe: (1, embed_dim, feat_size[0], feat_size[1])
            key_pe = get_feat_pe(self.pos_add_type, embed_dim, (feat_size,feat_size))
            key_pe = key_pe.flatten(2).permute(0, 2, 1).to(self.device)

        feat_logits = self.conv_fc(img_tokens)  # (bs, num_tokens, 1)

        attn_array = []
        for layer in self.layers:
            queries, img_tokens, attn_out_q = layer(
                queries=queries,
                keys=img_tokens,
                key_pe=key_pe,
            )
            # attn_out_q: (bs, num_heads, num_cls, L)
            # attn_score: (bs, num_cls, L)
            attn_score = torch.mean(attn_out_q, dim=1)
            attn_array.append(attn_score)
        
        # queries: (bs, n_cls, dim), img_tokens: (bs, num_tokens, dim)
        attn_map = torch.bmm(img_tokens, queries.transpose(1, 2))   # (bs, num_tokens, n_cls)
        # attn_map = attn_map / math.sqrt(embed_dim)
        attn_array.append(attn_map.transpose(1, 2))
        attn_array = torch.stack(attn_array, dim=1)
        return feat_logits, attn_map, attn_array
    
    
    def create_feat_gt(self, logits_shape, image_labels, clsid_mask):
        bs,num_tokens,_ = logits_shape
        # drop_neg_ratio = 0.1
        # keep_neg_nums = int(num_tokens*drop_neg_ratio)
        keep_neg_nums = 100
        feat_gt = torch.zeros((bs, num_tokens)).to(self.device)
        balanced_mask = torch.zeros((bs, num_tokens)).to(self.device)
        for img_label,cmask,bidx in zip(image_labels,clsid_mask,range(bs)):
            if img_label == 0:
                rand_keep = torch.randint(0, num_tokens, (keep_neg_nums,))
                balanced_mask[bidx, rand_keep] = 1
            else:
                feat_gt[bidx, cmask>0] = 1
                balanced_mask[bidx, cmask>0] = 1

        return feat_gt,balanced_mask
    
    def calc_loss(self,feature_emb, databatch):
        loss_fn_1 = nn.BCEWithLogitsLoss(reduction='none')
        loss_fn_2 = nn.CrossEntropyLoss(reduction='none')

        feat_logits, attn_map, _ = self.calc_logits(feature_emb)
        bs, num_tokens, _ = feat_logits.shape
        feat_size = int(math.sqrt(num_tokens))
        clsid_mask = F.interpolate(databatch['clsid_mask'].unsqueeze(1).float(), size=(feat_size,feat_size), mode='nearest')
        clsid_mask = clsid_mask.flatten(1).long().to(self.device)    # (bs, num_tokens)
        
        feat_gt,balanced_mask = self.create_feat_gt(feat_logits.shape, databatch['image_labels'], clsid_mask)
        loss_per_token = loss_fn_1(feat_logits.view(bs * num_tokens), feat_gt.view(bs * num_tokens))
        loss_per_token = loss_per_token.view(bs, num_tokens)
        feat_loss = loss_per_token * balanced_mask
        feat_loss = feat_loss.sum() / balanced_mask.sum().float()

        cls_loss = loss_fn_2(attn_map.permute(0, 2, 1), clsid_mask)
        cls_loss = cls_loss * balanced_mask
        cls_loss = cls_loss.sum() / balanced_mask.sum().float()
        loss = feat_loss + cls_loss
        return loss

    def set_pred(self,feature_emb, databatch):
        feat_logits,attn_map, attn_array = self.calc_logits(feature_emb) # (bs, num_classes)
        attn_map = F.softmax(attn_map, dim=-1)  # (bs, num_tokens, n_cls)
        # max_probs 的 shape 是 (bs, num_tokens)，即每个位置的预测概率
        # predicted_classes 的 shape 是 (bs, num_tokens)，即每个位置的预测类别索引
        max_probs, predicted_classes = torch.max(attn_map, dim=-1)
        databatch['token_probs'] = max_probs   # (bs, num_tokens)
        databatch['token_classes'] = predicted_classes # (bs, num_tokens)
        return databatch
