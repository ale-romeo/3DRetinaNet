from modules.anchor_box_retinanet import anchorBox as RanchorBox
from modules.anchor_box_kmeans import anchorBox as KanchorBox
from modules.detection_loss import FocalLoss
from models.backbone_models import backbone_models
from modules.box_utils import decode
import torch
import math
import torch.nn as nn
import modules.utils as utils
from models.cem_head import CEMHead  # NEW

logger = utils.get_logger(__name__)

class RetinaNet(nn.Module):
    def __init__(self, backbone, args):
        super(RetinaNet, self).__init__()

        self.num_classes = args.num_classes

        if args.ANCHOR_TYPE == 'RETINA':
            self.anchors = RanchorBox()
        elif args.ANCHOR_TYPE == 'KMEANS':
            self.anchors = KanchorBox()
        else:
            raise RuntimeError('Define correct anchor type')

        self.ar = self.anchors.ar
        args.ar = self.ar
        self.use_bias = True
        self.head_size = args.head_size
        self.backbone = backbone
        self.SEQ_LEN = args.SEQ_LEN
        self.HEAD_LAYERS = args.HEAD_LAYERS
        self.NUM_FEATURE_MAPS = args.NUM_FEATURE_MAPS

        self.reg_heads = self.make_head(self.ar * 4, args.REG_HEAD_TIME_SIZE, self.HEAD_LAYERS)
        self.cls_heads = self.make_head(self.ar * self.num_classes, args.CLS_HEAD_TIME_SIZE, self.HEAD_LAYERS)

        self.prior_prob = 0.01
        bias_value = -math.log((1 - self.prior_prob) / self.prior_prob)
        nn.init.constant_(self.cls_heads[-1].bias, bias_value)

        if args.MODE == 'train':
            self.criterion = FocalLoss(args)

        self.ego_head = nn.Conv3d(self.head_size, args.num_ego_classes, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
        nn.init.constant_(self.ego_head.bias, bias_value)

        self.use_cem = getattr(args, 'USE_CEM', False)
        if self.use_cem:
            self.cem_head = CEMHead(input_dim=self.head_size, concept_dim=args.num_concepts)
            self.cem_loss_fn = nn.BCELoss()

    def forward(self, images, gt_boxes=None, gt_labels=None, ego_labels=None, counts=None, img_indexs=None, concept_labels=None, get_features=False):
        sources, ego_feat = self.backbone(images)

        ego_preds = self.ego_head(ego_feat).squeeze(-1).squeeze(-1).permute(0, 2, 1).contiguous()

        if self.use_cem:
            concept_preds = self.cem_head(ego_feat)  # [B, T, num_concepts]

        grid_sizes = [feature_map.shape[-2:] for feature_map in sources]
        ancohor_boxes = self.anchors(grid_sizes)

        loc = list()
        conf = list()

        for x in sources:
            loc.append(self.reg_heads(x).permute(0, 2, 3, 4, 1).contiguous())
            conf.append(self.cls_heads(x).permute(0, 2, 3, 4, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), o.size(1), -1) for o in loc], 2)
        conf = torch.cat([o.view(o.size(0), o.size(1), -1) for o in conf], 2)

        flat_loc = loc.view(loc.size(0), loc.size(1), -1, 4)
        flat_conf = conf.view(conf.size(0), conf.size(1), -1, self.num_classes)

        if get_features:
            return flat_conf, sources
        elif gt_boxes is not None:
            total_loss = self.criterion(flat_conf, flat_loc, gt_boxes, gt_labels, counts, ancohor_boxes, ego_preds, ego_labels)
            if self.use_cem and concept_labels is not None:
                if concept_preds.shape != concept_labels.shape:
                    print(f"[CEM Debug] Shape mismatch: preds {concept_preds.shape}, labels {concept_labels.shape}")
                if concept_preds.device != concept_labels.device:
                    print(f"[CEM Debug] Device mismatch: preds on {concept_preds.device}, labels on {concept_labels.device}")
                #print(f"[CEM Debug] preds min/max: {concept_preds.min().item()} / {concept_preds.max().item()}")
                #print(f"[CEM Debug] labels sum: {concept_labels.sum().item()}")
                
                cem_loss = self.cem_loss_fn(concept_preds, concept_labels)
                total_loss = (total_loss[0], total_loss[1] + cem_loss)
            return total_loss
        else:
            decoded_boxes = []
            for b in range(flat_loc.shape[0]):
                temp_l = []
                for s in range(flat_loc.shape[1]):
                    temp_l.append(decode(flat_loc[b, s], ancohor_boxes))
                decoded_boxes.append(torch.stack(temp_l, 0))
            if self.use_cem:
                return torch.stack(decoded_boxes, 0), flat_conf, ego_preds, concept_preds
            else:
                return torch.stack(decoded_boxes, 0), flat_conf, ego_preds

    def make_features(self, shared_heads):
        layers = []
        use_bias = self.use_bias
        head_size = self.head_size
        for _ in range(shared_heads):
            layers.append(nn.Conv3d(head_size, head_size, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1), bias=use_bias))
            layers.append(nn.ReLU(True))

        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers

    def make_head(self, out_planes, time_kernel, num_heads_layers):
        layers = []
        use_bias = self.use_bias
        head_size = self.head_size

        for kk in range(num_heads_layers):
            branch_kernel = 1
            bpad = 0
            layers.append(nn.Conv3d(head_size, head_size, kernel_size=(branch_kernel, 3, 3), stride=1, padding=(bpad, 1, 1), bias=use_bias))
            layers.append(nn.ReLU(True))

        tpad = time_kernel // 2
        layers.append(nn.Conv3d(head_size, out_planes, kernel_size=(time_kernel, 3, 3), stride=1, padding=(tpad, 1, 1)))

        layers = nn.Sequential(*layers)

        for m in layers.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m.bias, 'data'):
                    nn.init.constant_(m.bias, 0)

        return layers

def build_retinanet(args):
    return RetinaNet(backbone_models(args), args)
