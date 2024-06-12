import os
import gc
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import pytorch_lightning as L
import timm

from glob import glob
from tqdm.auto import tqdm
from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from torchvision.io import read_image
from torchvision.transforms import v2 as  transforms
from torch.utils.data import Dataset, DataLoader
from transformers import ConvNextModel, Swinv2Model
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse

# pip install --quiet timm pytorch_lightning==1.7.7 torchmetrics==0.11.1

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default=None)

args = parser.parse_args()



class CustomDataset(Dataset):
    def __init__(self, df, path_col,  mode='train'):
        self.df = df
        self.path_col = path_col
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.mode == 'train':
            row = self.df.iloc[idx]
            image = read_image(row[self.path_col])/256.
            label = row['class']
            data = {
                'image':image,
                'label':label
            }
            return data
        elif self.mode == 'val':
            row = self.df.iloc[idx]
            image = read_image(row[self.path_col])/256.
            label = row['class']
            data = {
                'image':image,
                'label':label
            }
            return data
        elif self.mode == 'inference':
            row = self.df.iloc[idx]
            image = read_image(row[self.path_col])/256.
            data = {
                'image':image,
            }
            return data

    def train_transform(self, image):
        pass

class CustomCollateFn:
    def __init__(self, transform, mode):
        self.mode = mode
        self.transform = transform

    def __call__(self, batch):
        if self.mode=='train':
            pixel_values = torch.stack([self.transform(data['image']) for data in batch])
            label = torch.LongTensor([data['label'] for data in batch])
            return {
                'pixel_values':pixel_values,
                'label':label,
            }
        elif self.mode=='val':
            pixel_values = torch.stack([self.transform(data['image']) for data in batch])
            label = torch.LongTensor([data['label'] for data in batch])
            return {
                'pixel_values':pixel_values,
                'label':label,
            }
        elif self.mode=='inference':
            pixel_values = torch.stack([self.transform(data['image']) for data in batch])
            return {
                'pixel_values':pixel_values,
            }

class CustomModel(nn.Module):
    def __init__(self, model):
        super(CustomModel, self).__init__()
        self.model = model
        self.clf = nn.Sequential(
            nn.Tanh(),
            nn.LazyLinear(25),
        )

#     @torch.compile
    def forward(self, x, label=None):
        x = self.model(x)#.pooler_output
        x = self.clf(x)
        loss = None
        if label is not None:
            loss = nn.CrossEntropyLoss()(x, label)
        probs = nn.LogSoftmax(dim=-1)(x)
        return probs, loss

class LitCustomModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = CustomModel(model)
        self.validation_step_output = []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=1e-5)
        return opt

    def training_step(self, batch, batch_idx=None):
        x = batch['pixel_values']
        label = batch['label']
        probs, loss = self.model(x, label)
        self.log(f"train_loss", loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx=None):
        x = batch['pixel_values']
        label = batch['label']
        probs, loss = self.model(x, label)
        self.validation_step_output.append([probs,label])
        return loss

    def predict_step(self, batch, batch_idx=None):
        x = batch['pixel_values']
        probs, _ = self.model(x)
        return probs

    def validation_epoch_end(self, step_output):
        pred = torch.cat([x for x, _ in self.validation_step_output]).cpu().detach().numpy().argmax(1)
        label = torch.cat([label for _, label in self.validation_step_output]).cpu().detach().numpy()
        score = f1_score(label,pred, average='macro')
        self.log("val_score", score)
        self.validation_step_output.clear()
        return score


SEED = 40
N_SPLIT = 5
BATCH_SIZE = 12
L.seed_everything(SEED)
train_df = pd.read_csv('./data/train.csv')

paths = pd.Series(os.listdir("./data/srgan_10_train"))
train_df['img_path'] = paths.apply(lambda x: os.path.join('./data/srgan_10_train/',x))

le = LabelEncoder()
train_df['class'] = le.fit_transform(train_df['label'])

if not len(train_df) == len(os.listdir('./data/train')):
    raise ValueError()
skf = StratifiedKFold(n_splits=N_SPLIT, random_state=SEED, shuffle=True)
train_transform = transforms.Compose([
    transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])
val_transform = transforms.Compose([
    transforms.Resize(size=(224,224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

train_collate_fn = CustomCollateFn(train_transform, 'train')
val_collate_fn = CustomCollateFn(val_transform, 'val')

for fold_idx, (train_index, val_index) in enumerate(skf.split(train_df, train_df['class'])):
    #fold_idx += 4
    train_fold_df = train_df.loc[train_index,:]
    val_fold_df = train_df.loc[val_index,:]

    train_dataset = CustomDataset(train_fold_df, 'img_path', mode='train')
    val_dataset = CustomDataset(val_fold_df, 'img_path', mode='val')

    train_dataloader = DataLoader(train_dataset, collate_fn=train_collate_fn, batch_size=BATCH_SIZE,)
    val_dataloader = DataLoader(val_dataset, collate_fn=val_collate_fn, batch_size=BATCH_SIZE*2,)

    model = timm.create_model("timm/caformer_b36.sail_in22k_ft_in1k", pretrained=True, num_classes=25)
    lit_model = LitCustomModel.load_from_checkpoint(args.checkpoint, model = model)

  

    checkpoint_callback = ModelCheckpoint(
        monitor='val_score',
        mode='max',
        dirpath='./checkpoints/',
        filename=f'caformer-resize-fold_idx={fold_idx}'+'-{epoch:02d}-{train_loss:.4f}-{val_score:.4f}',
        save_top_k=1,
        save_weights_only=True,
        verbose=True
    )
    earlystopping_callback = EarlyStopping(monitor="val_score", mode="max", patience=3,check_on_train_epoch_end=False)
    trainer = L.Trainer(max_epochs=100, accelerator='auto' ,precision=32, callbacks=[checkpoint_callback, earlystopping_callback], val_check_interval=0.5,num_nodes=1)
    trainer.fit(lit_model, train_dataloader, val_dataloader)

    model.cpu()
    lit_model.cpu()
    del model, lit_model, checkpoint_callback, earlystopping_callback, trainer
    gc.collect()
    torch.cuda.empty_cache()




