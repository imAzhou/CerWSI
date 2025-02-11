import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class CerMClassifier(nn.Module):
    def __init__(self, num_classes, num_patches, embed_dim):
        super(CerMClassifier, self).__init__()
       
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.cls_neg_head = nn.Linear(self.embed_dim + (num_classes-1), 1)
        self.cls_pos_heads = nn.ModuleList()
        for i in range(num_classes-1):
            self.cls_pos_heads.append(nn.Linear(num_patches, 1))
        
        self.drop_neg_ratio = 0.5 # 样本平衡因子，阴性token丢弃比例
        self.feat_loss_factor = 0.5
        self.token_linear_head = nn.Linear(self.embed_dim, 1)
    
    @property
    def device(self):
        return next(self.parameters()).device

    def calc_logits(self, feature_emb: torch.Tensor):
        '''
        Args:
            feature_emb: (bs,cls_token+img_token,C)
        Return:
            img_logits: (bs, 1)
            token_logits: (bs, num_tokens, 1)
        '''
        cls_neg_token = feature_emb[:,0,:]  # (bs, C)
        cls_pos_tokens = feature_emb[:,1:self.num_classes,:]  # (bs, num_cls-1, C)
        img_tokens = feature_emb[:,self.num_classes:,:]

        simi_matrix = torch.matmul(cls_pos_tokens, img_tokens.transpose(1, 2))  # (bs, num_cls-1, num_tokens)
        simi_matrix = (simi_matrix - simi_matrix.mean(-1, keepdim=True)) / (simi_matrix.std(-1, keepdim=True) + 1e-8)

        avg_pos_token = torch.mean(simi_matrix, dim=-1)
        overall_neg_token = torch.cat([cls_neg_token,avg_pos_token], dim=-1 )
        pred_img_neg_logits = self.cls_neg_head(overall_neg_token)  # (bs, 1)

        pred_img_pos_logits = []
        for i in range(self.num_classes-1):
            pred_img_pos_logits.append(self.cls_pos_heads[i](simi_matrix[:,i,:]))  # [(bs, 1),]
        pred_img_pos_logits = torch.cat(pred_img_pos_logits, dim=-1)  # (bs, num_cls-1)

        token_logits = self.token_linear_head(img_tokens) # (bs, num_tokens, 1)

        return pred_img_neg_logits, pred_img_pos_logits, token_logits
    
    def calc_loss(self,pred_logits, databatch):
        img_neg_logits,img_pos_logits,token_logits = pred_logits
        device = img_neg_logits.device
        img_gt = databatch['image_labels'].to(device).unsqueeze(-1).float()
        img_neg_loss = F.binary_cross_entropy_with_logits(img_neg_logits, img_gt, reduction='mean')
        token_loss = self.calc_feat_loss(token_logits, databatch)
        img_pos_loss = self.calc_pos_loss(img_pos_logits, databatch)
        loss = img_neg_loss + img_pos_loss + token_loss
        return loss

    def create_feat_gt(self, logits_shape, databatch):
        bs,num_tokens,_ = logits_shape
        keep_neg_nums = int(num_tokens*self.drop_neg_ratio)
        feat_gt = torch.zeros((bs, num_tokens)).to(self.device)
        balanced_mask = torch.zeros((bs, num_tokens)).to(self.device)
        feat_hw = int(math.sqrt(num_tokens))
        for img_label,token_labels,bidx in zip(databatch['image_labels'],databatch['token_labels'],range(bs)):
            if len(token_labels) > 0:
                for row,col,clsid in token_labels:
                    # 对于阳性图片来说，标记了类别的token记为1，其他为0的只是没标记，不代表没病变
                    feat_gt[bidx, (row*feat_hw)+col] = 1
                    balanced_mask[bidx, (row*feat_hw)+col] = 1
            if img_label == 0:
                # 对于阴性图片来说，它所有 token 对于类别0来说都是正样本
                feat_gt[bidx, :] = 0
                rand_keep = torch.randint(0, num_tokens, (keep_neg_nums,))
                balanced_mask[bidx, rand_keep] = 1

        return feat_gt,balanced_mask

    def calc_feat_loss(self, feat_logits, databatch):
        bs, num_tokens, _ = feat_logits.shape
        feat_gt,balanced_mask = self.create_feat_gt(feat_logits.shape, databatch)
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        loss_per_token = criterion(feat_logits.view(bs * num_tokens), feat_gt.view(bs * num_tokens))
        loss_per_token = loss_per_token.view(bs, num_tokens)
        masked_loss = loss_per_token * balanced_mask
        # 计算所有类别的损失，按掩码取有效位置的损失并计算平均值
        final_clshead_loss = masked_loss.sum() / balanced_mask.sum().float()
        
        # img_tokens: (bs, num_tokens, C)   feat_gt/balanced_mask:(bs, num_tokens)
        # indices = balanced_mask.nonzero(as_tuple=True)
        # contrast_feat = img_tokens[indices]
        # contrast_feat = F.normalize(contrast_feat, dim=-1)  # bs*num_tokens, c
        # token_gt = feat_gt[indices]
        # cont_loss = contrastive_loss(contrast_feat, token_gt, self.temperature)
        # token_loss = cont_loss + final_clshead_loss
        
        return final_clshead_loss

    def calc_pos_loss(self, pos_logits, databatch):
        loss_fn = nn.BCEWithLogitsLoss()
        binary_matrix = torch.zeros_like(pos_logits, dtype=torch.float32)
        for i, token_labels in enumerate(databatch['token_labels']):
            label_list = list(set([tk[-1] -1 for tk in token_labels]))  # GT阳性类别id范围为 [1,5], pred阳性类别id范围为 [0,4]
            binary_matrix[i, label_list] = 1

        loss = loss_fn(pos_logits, binary_matrix)
        return loss
