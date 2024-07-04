#  A super-resolution framework for license plate recognition
## Project
### Setting up
* please create a conda environment
* install the packages according to requirements.txt
* please clone the project and download the proposed dataset in google drive (HR_782.zip, LR_782.zip)
### Training
* define the .yml file under ./options/train/SwinFIR
    * name of the experiment
    * dataset directory
    * scheduler, total_iter
    * loss functions
* run the command (e.g. below)
```
CUDA_VISIBLE_DEVICES=0 python swinfir/train.py -opt options/train/SwinFIR/train_crnn4_Relu_VGG.yml
```
* models and training_states will be saved under ./experiments
### Tensorboard
* watch the PSNR/SSIM curves and loss value curves during validation
```
tensorboard --logdir tb_logger/crnnOCR_continue --port 5500 --bind_all
```
### Testing
* modify the .yml file ./options/test/SwinFIR/SwinFIR_SRx2.yml
    * name of testing
    * dataset directory
    * model path (pretrain_network_g)
* the output super-resolution images will be saved under ./results

## OCR recognizers
### Multi-task
* the OCR model is download from https://github.com/Valfride/lpr-rsr-ext
* please refer to PKU-SR.zip in google drive
* run OCRpred_SR.py to get the recognition results from Multi-task
### PaddleOCR
* please clone the project https://github.com/PaddlePaddle/PaddleOCR
* please put model file under ./models
    * please refer to ch_PP-OCRv3_rec_train.zip in google drive
* modify /configs/ch_PP-OCRv3_rec.yml
    * change infer_img
* modify /tools/infer_rec.py
    * change output_file_path
* run the command
```
python3 tools/infer_rec.py -c configs/ch_PP-OCRv3_rec.yml
```
* run calculate_acc.py to output .csv file
    * change the input txt path and output filename
### CRNN
* please clone the project from https://github.com/we0091234/crnn_plate_recognition/tree/master
* the model is already under ./saved_model
* please modify demo.py
    * change image_path and output text file path
* run calculate_acc.py to output .csv file
    * change the input txt path and output filename

## Dataset
### Proposed dataset (in google drive)
* HR_782 and LR_782 are used for our project
* LR_bicubic is for showing LR image with the same size of HR image, it has been resized using bicubic method to x2
* HR_ori_size and LR_ori_size : each image is in their original size
* LSVLP_cropped_beforePS : each image is unrectified by Photoshop

### PKU-SR dataset (in google drive)
* original PKU-SR dataset is PKU-Dataset-SR.zip
    * clone the project https://github.com/Valfride/lpr-rsr-ext to train and test on it
* PKUSR.zip is for training and testing our method
    * data has been split to train/val/test folder
### Others (in google drive)
* FSI-DI-Dataset is the original dataset from the paper https://www.sciencedirect.com/science/article/pii/S2666281720303899
* FSI-DI : HR-LR paired images which are collected from FSI-DI-Dataset
