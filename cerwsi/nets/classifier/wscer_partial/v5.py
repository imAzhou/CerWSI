import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..meta_classifier import MetaClassifier
from cerwsi.utils import build_evaluator,ImgODMetric
from .binary_cls_branch import BinaryClsBranch
from .instance_branch import Instance_branch


class WSCerPartial(MetaClassifier):
    def __init__(self, args):

        save_result_dir = getattr(args, 'save_result_dir', None)
        evaluator = build_evaluator([ImgODMetric(args.logger_name,save_result_dir)])
        super(WSCerPartial, self).__init__(evaluator, **args)

        self.binary_cls_branch = BinaryClsBranch(args.binary_branch_input_dim)
        self.instance_branch = Instance_branch(
            transformer_dim = 256,
            img_input_size = args.img_size,
            num_classes = args.num_classes,
            num_instance_queries = args.num_instance_queries,
            pretrain_ckpt = args.instance_ckpt)
        
    def calc_loss(self, dict_inputs: dict, databatch):
        '''
        dict_inputs: dict, 
            vision_features: Tensor, (bs, c, h, w)
            vision_pos_enc: List[Tensor]: [bs, c, h1,w1]...
            backbone_fpn: List[Tensor]: [bs, c, h1,w1]...
        '''
        img_logits,binary_attnmap = self.binary_cls_branch(dict_inputs['vision_features'])
        binary_loss_fn = nn.BCEWithLogitsLoss()
        img_gt = databatch['image_labels'].unsqueeze(1).float()
        img_loss = binary_loss_fn(img_logits, img_gt)
        
        bs,num_tokens = binary_attnmap.shape
        feat_size = int(math.sqrt(num_tokens))
        binary_attnmap = binary_attnmap.reshape((bs, feat_size, feat_size)).unsqueeze(1)
        binary_attnmap = F.interpolate(binary_attnmap, size=(feat_size*4, feat_size*4), mode='nearest')
        
        instance_loss_dict = self.instance_branch.loss(dict_inputs, databatch, binary_attnmap)
        loss = img_loss
        loss_dict = {'img_loss': img_loss.item()}
        for key,value in instance_loss_dict.items():
            loss += value
            loss_dict[key] = value.item()

        return loss, loss_dict
    
    def set_pred(self, dict_inputs, databatch):
        img_logits,binary_attnmap = self.binary_cls_branch(dict_inputs['vision_features'])
        databatch['img_probs'] = torch.sigmoid(img_logits)
        
        bs,num_tokens = binary_attnmap.shape
        feat_size = int(math.sqrt(num_tokens))
        binary_attnmap = binary_attnmap.reshape((bs, feat_size, feat_size)).unsqueeze(1)
        databatch['binary_attnmap'] = binary_attnmap
        binary_attnmap = F.interpolate(binary_attnmap, size=(feat_size*4, feat_size*4), mode='nearest')
        databatch['pred_bbox'] = self.instance_branch.predict(dict_inputs, databatch, binary_attnmap)
        return databatch
