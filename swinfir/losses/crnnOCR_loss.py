import torch
from torch import nn as nn
from torch.nn import functional as F
from ..archs.plateNet_Extract import myNet_ocr
from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.loss_util import weighted_loss
from pathlib import Path

device = torch.device("cuda")
path_ocr = Path('./swinfir/crnn_model/best.pth')
# model_path = "../crnn_model/best.pth"

plate_chr="#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"

def init_model(device,model_path, extract_layers):
    check_point = torch.load(model_path,map_location=device)
    model_state=check_point['state_dict']
    cfg = check_point['cfg']
    model = myNet_ocr(num_classes=len(plate_chr),export=True,cfg=cfg, extract_layers=extract_layers)       
    model.load_state_dict(model_state)
    model.to(device)
    model.eval()
    return model

@LOSS_REGISTRY.register()
class crnnOCR_Loss(nn.Module):
    """Perceptual loss.
    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 use_input_norm=True,
                 range_norm=True,
                 perceptual_weight=1.0,
                 criterion='l1'):
        super(crnnOCR_Loss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.layer_weights = layer_weights

        # self.swin = SwinTFeatureExtractor(
        #     layer_name_list=list(layer_weights.keys()),
        #     use_input_norm=use_input_norm,
        #     range_norm=range_norm)
        self.model = init_model(device,path_ocr,list(layer_weights.keys()))

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """

        # out, features = model(x)

        # extract ocr features
        # _ , x_features = self.model(x)
        # _ , gt_features = self.model(gt.detach())

        x_features = self.model(x)
        gt_features = self.model(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            # percep_loss = 0
            ocr_percep_loss = torch.tensor(0.0, dtype=torch.float32, device=x.device)
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    ocr_percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    ocr_percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            ocr_percep_loss *= self.perceptual_weight
        else:
            ocr_percep_loss = None

        style_loss = None

        return ocr_percep_loss , style_loss