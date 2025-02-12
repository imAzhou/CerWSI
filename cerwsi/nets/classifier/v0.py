import torch
import torch.nn as nn
import torch.nn.functional as F

class CerMClassifier(nn.Module):
    def __init__(self, num_classes, num_patches, embed_dim):
        super(CerMClassifier, self).__init__()
       
        self.embed_dim = embed_dim
        self.cls_neg_head = nn.Linear(self.embed_dim, 1)
        
    @property
    def device(self):
        return next(self.parameters()).device

    def calc_logits(self, feature_emb: torch.Tensor):
        '''
        Args:
            feature_emb: (bs,cls_token+img_token,C)
        Return:
            img_logits: (bs, 1)
        '''
        cls_neg_token = feature_emb[:,0,:]  # (bs, C)
        pred_img_neg_logits = self.cls_neg_head(cls_neg_token)  # (bs, 1)
        return pred_img_neg_logits
    
    def calc_loss(self,feature_emb, databatch):
        img_neg_logits = self.calc_logits(feature_emb)
        img_gt = databatch['image_labels'].to(self.device).unsqueeze(-1).float()
        img_neg_loss = F.binary_cross_entropy_with_logits(img_neg_logits, img_gt, reduction='mean')
        return img_neg_loss

    def set_pred(self,feature_emb, databatch):
        img_neg_logits = self.calc_logits(feature_emb)
        databatch['img_probs'] = torch.sigmoid(img_neg_logits).squeeze(-1)   # (bs, )
        return databatch
