import torch
import torch.nn as nn
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
        
        filter_dict_inputs, filter_databatch, filter_binary_attnmap = self.filter4instance(
            dict_inputs, databatch, binary_attnmap
        )
        instance_loss_dict = self.instance_branch.loss(
            filter_dict_inputs, filter_databatch, filter_binary_attnmap)
        loss = img_loss
        loss_dict = {'img_loss': img_loss.item()}
        for key,value in instance_loss_dict.items():
            loss += value
            loss_dict[key] = value.item()

        return loss, loss_dict
     
    
    def set_pred(self, dict_inputs, databatch):
        img_logits,binary_attnmap = self.binary_cls_branch(dict_inputs['vision_features'])
        databatch['img_probs'] = torch.sigmoid(img_logits)
        databatch['binary_attnmap'] = binary_attnmap
        filter_dict_inputs, filter_databatch, filter_binary_attnmap = self.filter4instance(
            dict_inputs, databatch, binary_attnmap
        )
        databatch['pred_bbox'] = self.instance_branch.predict(
            filter_dict_inputs, filter_databatch, filter_binary_attnmap)
        return databatch

    def filter4instance(self, dict_inputs, databatch, binary_attnmap):
        new_dict_inputs, new_databatch, new_binary_attnmap = [],[],[]
        for input,dataitem,attnmap in zip(dict_inputs, databatch, binary_attnmap):
            if dataitem['use_inst']:
                new_dict_inputs.append(input)
                new_databatch.append(dataitem)
                new_binary_attnmap.append(attnmap)
        new_dict_inputs = torch.stack(new_dict_inputs, dim=0)
        new_binary_attnmap = torch.stack(new_binary_attnmap, dim=0)
        
        return new_dict_inputs,new_databatch,new_binary_attnmap