# TransDoubleNet: Semantic segmentation

## Description
All commands and setup below is required for all three models. For transUnet and transUnetDoubbleEncoder, there are additional setup instructions.

Setup the Environment

1. [Install CUDA](https://developer.nvidia.com/cuda-downloads)

2. [Install PyTorch 1.13 or later](https://pytorch.org/get-started/locally/)

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download the data and run training:
[All Processed Images](https://drive.google.com/file/d/1wd1ziFTxH41VOPb_fA_xctLkvw8OPUB8/view?usp=drive_link)

Select desired masks and put them in masks_train and masks_test inside data folder accordingly.

6. Download Pretrained Models (Optional)
[All Pretrained Models](https://drive.google.com/file/d/1rLx_zKlYOATkLVVHVGPARX_bCr7mgokh/view?usp=drive_link)

Select desired pretrained models based on tasks. Specified the path of pretrained models for prediction.


## Usage
**Note : Use Python 3.6 or newer**

### Training

```
> python train.py -h
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --validation VAL, -v VAL
                        Percent of the data that is used as validation (0-100)
  --amp                 Use mixed precision
```

Automatic mixed precision is also available with the `--amp` flag. [Mixed precision](https://arxiv.org/abs/1710.03740) allows the model to use less memory and to be faster on recent GPUs by using FP16 arithmetic. Enabling AMP is recommended.

To train based on our developed settings, use the following command:

`python train.py -e 1500 -l 0.0000005 -b 2 -c 1 --amp`

### Prediction

After training your model and saving it to `MODEL.pth`, you can easily test the output masks on your images via the CLI.

To predict a single image and save it:

`python predict.py -i image.jpg -o output.jpg`

To predict a multiple images and show them without saving them:

`python predict.py -i image1.jpg image2.jpg --viz --no-save`

```console
> python predict.py -h
usage: predict.py [--model FILE] --input INPUT [INPUT ...] 
                  [--output INPUT [INPUT ...]] [--viz] [--no-save]
                  [-c]

Predict masks from input images

optional arguments:
  --model FILE, -m FILE
                        Specify the file in which the model is stored
  --input INPUT [INPUT ...], -i INPUT [INPUT ...]
                        Filenames of input images
  --output INPUT [INPUT ...], -o INPUT [INPUT ...]
                        Filenames of output images
  --viz, -v             Visualize the images as they are processed
  --no-save, -n         Do not save the output masks
  -c                    Specify the number of classes
```
You can specify which model file to use with `--model MODEL.pth`.

### Reference
[UNet Implementation](https://github.com/milesial/Pytorch-UNet)
[TransUnet Model](https://github.com/Beckschen/TransUNet)

## Data
Unprocessed Original Data can be found here:
[Unprocessed Data](https://drive.google.com/file/d/1cBfQjVoD0U--ckFnTwiVLJPKYo7OiJ4I/view?usp=drive_link)
