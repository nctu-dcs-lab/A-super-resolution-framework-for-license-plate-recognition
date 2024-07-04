import torch
from torch import nn as nn
from torchvision.transforms.functional import normalize
from basicsr.utils.registry import METRIC_REGISTRY
import lpips

@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, crop_border, input_order='HWC', test_y_channel=False, **kwargs):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity).

    Args:
        img (ndarray): Images with range [0, 1].
        img2 (ndarray): Images with range [0, 1].
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        input_order (str): Whether the input order is 'HWC' or 'CHW'. Default: 'HWC'.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: LPIPS result.
    """

    
    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')
    
    # Reorder images if necessary
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)
    
    # Convert to tensors
    img_tensor = torch.tensor(img.transpose(2, 0, 1)).unsqueeze(0)
    img2_tensor = torch.tensor(img2.transpose(2, 0, 1)).unsqueeze(0)
    
    # Normalize images
    img_tensor = normalize(img_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    img2_tensor = normalize(img2_tensor, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    
    # Calculate LPIPS using lpips module
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # Assuming GPU is available
    lpips_val = loss_fn_vgg(img_tensor, img2_tensor)
    
    return lpips_val.item()  # Return LPIPS result as float