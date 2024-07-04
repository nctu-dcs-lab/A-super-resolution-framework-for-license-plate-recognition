import os.path as osp
import swinfir.archs
import swinfir.data
import swinfir.models
import swinfir.losses
from basicsr.train import train_pipeline

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)


# CUDA_VISIBLE_DEVICES=0 python swinfir/train.py -opt options/train/SwinFIR/train_SwinFIR_SRx2_from_scratch.yml --launcher pytorch


# CUDA_VISIBLE_DEVICES=0 python swinfir/train.py -opt options/train/SwinFIR/train_crnn4_Relu_VGG.yml
# CUDA_VISIBLE_DEVICES=0 python swinfir/train.py -opt options/train/SwinFIR/trainOnPKU_SwinT_DISTS.yml
# CUDA_VISIBLE_DEVICES=0 python swinfir/train.py -opt options/train/SwinFIR/train_crnn4_conv.yml --auto_resume
# # --auto_resume