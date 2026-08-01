import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.sglatrackv2.vit import vit_base_patch16_224, vit_large_patch16_224
from lib.models.sglatrackv2.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.models.sglatrackv2.vit_cae_async import CAE_Base_patch16_224_Async, CAE_Tiny_patch16_224_Async,CAE_Small_patch16_224_Async
from lib.utils.box_ops import box_xyxy_to_cxcywh


class SGLATrackV2(nn.Module):
    """ This is the base class for MMTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=1):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)
        
        # track query: save the history information of the previous frame
        self.track_query = None
        self.token_len = token_len

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                profile=False,
                ):
        # assert isinstance(search, list), "The type of search is not List"

        if profile:

            out_dict = []
            hlist = torch.tensor([])

            x, aux_dict = self.backbone(template, search,
                                            hlist=hlist, mode='zx')
            if hlist.numel() > 0:
                hlist = torch.cat( (hlist,aux_dict['hlist']) , dim=1 )
            else:
                hlist = aux_dict['hlist']

            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1]
                    
            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)

            opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
                # Forward head
            out = self.forward_head(opt, None)

            out.update(aux_dict)

            out_dict.append(out)
                
            return out_dict
        else:

            out_dict = []

            z, aux_dict = self.backbone(z=template.copy(), x=None,
                                        hlist=None, mode='z')
            
            B,_,dim = z.shape
            hlist = torch.zeros(B,6,dim).to(z.device)
            
            for i in range(len(search)):
                x, aux_dict = self.backbone(z, x=search[i],
                                            hlist=hlist, mode='zx')

                hlist = aux_dict['hlist']

                feat_last = x
                if isinstance(x, list):
                    feat_last = x[-1]
                    
                enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)

                opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
                # Forward head
                out = self.forward_head(opt, None)

                out.update(aux_dict)
                out['backbone_feat'] = x
                
                out_dict.append(out)
                
            return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        enc_opt: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()

        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            
            out = {'pred_boxes': outputs_coord_new,
                    'score_map': score_map_ctr,
                    'size_map': size_map,
                    'offset_map': offset_map}
            
            return out
        else:
            raise NotImplementedError

    def forward_test(self, template: torch.Tensor,
                search: torch.Tensor,
                mode,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                hlist = None
                ):

        if mode == 'z':
            z, aux_dict = self.backbone.forward_z(z=template.copy())
            return z, aux_dict
        else:

            out_dict = []
            for i in range(len(search)):
                x, aux_dict = self.backbone.forward_zx_test(template, x=search[i],
                                            hlist=hlist)
                feat_last = x
                if isinstance(x, list):
                    feat_last = x[-1]
                    
                enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)

                opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
                # Forward head
                out = self.forward_head(opt, None)

                out.update(aux_dict)
                out['backbone_feat'] = x
                
                out_dict.append(out)
                
            return out_dict, aux_dict


def build_sglatrackv2(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_networks')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,)

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE, 
                                         add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                         attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, 
                                         )
        
    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            )

    elif cfg.MODEL.BACKBONE.TYPE == 'caev2_tiny':
        backbone = CAE_Tiny_patch16_224_Async(pretrained,
                                              drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                              add_cls_token=False,
                                              num_async_interaction_stage=12,
                                              depth=12
                                              )
        
    elif cfg.MODEL.BACKBONE.TYPE == 'caev2_small':
        backbone = CAE_Small_patch16_224_Async(pretrained,
                                              drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                              add_cls_token=False,
                                              num_async_interaction_stage=12,
                                              depth=12
                                              )
    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1
    
    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = SGLATrackV2(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
    )

    return model

