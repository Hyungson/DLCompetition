import random
import pandas as pd
import numpy as np
import os
import re
import glob
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import v2 as  transforms

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torchvision.models as models
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from tqdm import tqdm

import warnings
warnings.filterwarnings(action='ignore') 

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

CFG = {
    'IMG_SIZE':256,
    #'EPOCHS':20,
    'LEARNING_RATE':3e-4,
    'BATCH_SIZE':24,
    'SEED':41
}

df = pd.read_csv('./train.csv')

df = pd.read_csv('train.csv')
paths = pd.Series(os.listdir("../srgan/data/sr_train"))
df['img_path'] = paths.apply(lambda x: os.path.join('../srgan/data/sr_train/',x))
train, val, _, _ = train_test_split(df, df['label'], test_size=0.3, stratify=df['label'], random_state=1999)


## Label-Encoding
le = preprocessing.LabelEncoder()
train['label'] = le.fit_transform(train['label'])
val['label'] = le.transform(val['label'])

class CustomDataset(Dataset):
    def __init__(self, img_path_list, label_list, transforms=None):
        self.img_path_list = img_path_list
        self.label_list = label_list
        self.transforms = transforms
        
    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        
        image = cv2.imread(img_path)
       
        
        if self.transforms is not None:
            image = self.transforms(image=image)['image']
        
        if self.label_list is not None:
            label = self.label_list[index]
            return image, label
        else:
            return image
        
    def __len__(self):
        return len(self.img_path_list)

# train_transform = transforms.Compose([
#     transforms.Resize(size=(256,256), interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
#     transforms.ToTensor()
# ])
# test_transform = transforms.Compose([
#     transforms.Resize(size=(256,256), interpolation=transforms.InterpolationMode.BICUBIC),
#     transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
#     transforms.ToTensor()
# ])

train_transform = A.Compose([
                            #A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE'],interpolation=cv2.INTER_LANCZOS4),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            #A.HorizontalFlip(),
                            ToTensorV2()
                            ])

test_transform = A.Compose([
                            #A.Resize(CFG['IMG_SIZE'],CFG['IMG_SIZE'], interpolation=cv2.INTER_LANCZOS4),
                            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0, always_apply=False, p=1.0),
                            #A.HorizontalFlip(),
                            ToTensorV2()
                            ])

train_dataset = CustomDataset(train['img_path'].values, train['label'].values, train_transform)
train_loader = DataLoader(train_dataset, batch_size = CFG['BATCH_SIZE'], shuffle=True, num_workers=0)

val_dataset = CustomDataset(val['img_path'].values, val['label'].values, test_transform)
val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)



test = pd.read_csv('./test.csv')
test_dataset = CustomDataset(test['img_path'].values, None, test_transform)
test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=0)