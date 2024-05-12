CNN.ipynb (Yintong Liu)  
DNN.ipynb (Yintong Liu)  
Download data from https://www.kaggle.com/datasets/devzohaib/dog-emotions-prediction and upload to google drive.  
Run the Notebook


# EmoNet
Go to this google drive: https://drive.google.com/drive/folders/1pEHSHE_T01FdzNPhr0RsD6_Dy1V9nRcy
And select and download the latest training weight and add it in the folder that contains “TEST_MODEL.py”

The latest weight currently is 10_85.pth

Training a model:
Go to main.py, line 117
 parser.add_argument('--pretrained_model', type=str, default='./snapshot/models/EmotionNet/normal/fold_0/Imagenet/11_16.pth')

Replace the path with the training weight that you downloaded: ‘.../Imagenet/XX_YY.pth’

In the terminal, run:
python main.py --image_size 256 --lr 0.001 --num_epochs 5 --batch_size 32 --fold 0 --mode train

Testing a model:
Go to main.py, line 117
 parser.add_argument('--pretrained_model', type=str, default='./snapshot/models/EmotionNet/normal/fold_0/Imagenet/11_16.pth')

Replace the path with the training weight that you downloaded: ‘.../Imagenet/XX_YY.pth’

In the terminal, run:
python main.py --image_size 256 --lr 0.001 --num_epochs 1 --batch_size 100 --fold 0 --mode test

The last 2 commented lines is also given in the main file for testing and training.

(For clearer instructions: https://docs.google.com/document/d/12_V4hhbh7AesxHiUVYgzSf_n1xCTAOHYU_P7H3V_mXk)

The rest of the files were adjusted from the original repository
(Eliot Yoon)
