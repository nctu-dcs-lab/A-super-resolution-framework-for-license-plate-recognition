import torch
from torch import nn
from torchvision.transforms.functional import normalize
from basicsr.utils.registry import LOSS_REGISTRY
from DISTS_pytorch import DISTS

from basicsr.losses.loss_util import weighted_loss

@weighted_loss
def dists_loss(x, gt):
    D = DISTS().cuda()
    # Calculate DISTS between x and gt with require_grad=True and batch_average=True
    dists_loss = D(x, gt, require_grad=True, batch_average=True)
    return dists_loss

@LOSS_REGISTRY.register()
class DISTS_Loss(nn.Module):
    """DISTS loss for perceptual loss calculation in super-resolution.

    Args:
        dists_weight (float): Weight for the DISTS loss. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0):
        super(DISTS_Loss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # Multiply the calculated dists_loss by the dists_weight
        return self.loss_weight * dists_loss(x.cuda(), gt.cuda())
