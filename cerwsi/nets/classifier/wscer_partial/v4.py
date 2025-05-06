import torch
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
        img_loss,_ = self.binary_cls_branch.loss(dict_inputs['vision_features'], databatch)
        instance_loss_dict = self.instance_branch.loss(dict_inputs, databatch)
        loss = img_loss
        loss_dict = {'img_loss': img_loss.item()}
        for key,value in instance_loss_dict.items():
            loss += value
            loss_dict[key] = value.item()

        return loss, loss_dict
     
    
    def set_pred(self,dict_inputs, databatch):
        img_logits, binary_attnmap = self.binary_cls_branch(dict_inputs['vision_features'])
        databatch['img_probs'] = torch.sigmoid(img_logits)
        databatch['binary_attnmap'] = binary_attnmap
        databatch['pred_bbox'] = self.instance_branch.predict(dict_inputs, databatch)
        return databatch
