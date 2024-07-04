import torch
from torch import nn as nn
from torch.nn import functional as F
from ..archs.Swin_arch import SwinTFeatureExtractor
from basicsr.utils.registry import LOSS_REGISTRY
from basicsr.losses.loss_util import weighted_loss

@LOSS_REGISTRY.register()
class Swin_PLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 use_input_norm=True,
                 range_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(Swin_PLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        # self.vgg = VGGFeatureExtractor(
        #     layer_name_list=list(layer_weights.keys()),
        #     vgg_type=vgg_type,
        #     use_input_norm=use_input_norm,
        #     range_norm=range_norm)


        self.swin = SwinTFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            use_input_norm=use_input_norm,
            range_norm=range_norm)

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
        # extract swin features
        x_features = self.swin(x)
        gt_features = self.swin(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            # percep_loss = 0
            percep_loss = torch.tensor(0.0, dtype=torch.float32, device=x.device)
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            # style_loss = 0
            style_loss = torch.tensor(0.0, dtype=torch.float32, device=x.device)
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None


        # print("(Original) Type of percep_loss:", type(percep_loss))
        # print("(Original) Type of style_loss:", type(style_loss))

        # # Convert to tensor if not None
        # if percep_loss is not None:
        #     percep_loss = torch.tensor(percep_loss)
        # if style_loss is not None:
        #     style_loss = torch.tensor(style_loss)

        

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram