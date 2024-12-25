import torch
from torch.utils.data import Dataset,DataLoader
import logging
from PIL import Image

import os
import numpy as np
import matplotlib as plt
from multiprocessing import Pool
from tqdm import tqdm
import torchvision.transforms as T
import random
from med_clip import CLIPModel
from CFG import CFG
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
normal_clip, clip_preprocess = clip.load("ViT-B/32", device=CFG.device)

import csv

def load_centroids_from_csv(filepath):
    centroids = {}
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for row in reader:
            filename, x, y = row
            centroids[filename] = (float(x), float(y))
    return centroids

class MultitaskDataset(Dataset):
    def __init__(self, image_folder,med_clip_path,image_folder_224, mask_folder,filenames, centroids_filepath,scale: float = 1.0):
        self.image_folder = image_folder
        self.image_folder_224 = image_folder_224
        self.mask_folder = mask_folder

        self.filenames = filenames
        self.scale = scale
        self.centroids = load_centroids_from_csv(centroids_filepath)
        self.med_clip_model = CLIPModel().to(CFG.device)
        self.med_clip_model.load_state_dict(torch.load(med_clip_path, map_location=CFG.device))
        self.med_clip_model.eval()

        #self.augment = augment

        #self.transforms = T.Compose([
            #T.RandomHorizontalFlip(p=0.5),
            #T.RandomVerticalFlip(p=0.5),
        #])

    def __len__(self):
        return len(self.filenames)

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            if img.ndim == 2:
                mask = np.where(img == 255, 1, 0)
                mask = np.expand_dims(mask, axis=0)

            else:
                assert 'dimension more than one'




            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    #@staticmethod
    def encode_image_with_clip(self,image):
        image_transform = clip_preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = self.med_clip_model.image_encoder(image_transform)
            image_embeddings = self.med_clip_model.image_projection(image_features)
        return image_embeddings.squeeze(0)

    @staticmethod
    def encode_centroids_with_clip(centroids):

        texts = [f"The breast center of this image at x={x:.2f}, y={y:.2f}" for x, y in centroids]

        text_tensors = clip.tokenize(texts).to(device)

        with torch.no_grad():
            text_features = normal_clip.encode_text(text_tensors)

        return text_features.squeeze(0)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image_path = os.path.join(self.image_folder, filename)
        image_path_224 = os.path.join(self.image_folder_224, filename)
        mask_path = os.path.join(self.mask_folder, filename)

        mask = Image.open(mask_path)
        img = Image.open(image_path)
        #if self.augment:
            #img = self.transforms(img)
            #mask = self.transforms(mask)


        img_224 = Image.open(image_path_224)  
        img_224_feature = self.encode_image_with_clip(img_224)
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask,self.scale,is_mask=True)
        type_label,sub_label = self.parse_labels_from_filename(image_path)
        centroid = self.centroids.get(filename, (0.0, 0.0))
        centroid_features = self.encode_centroids_with_clip([centroid])

        return {
            'idx':image_path,
            'mask_path':mask_path,
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'img_clip_feature':img_224_feature,
            'mask': torch.as_tensor(mask.copy()).long().contiguous(),  # Return mask as numpy array
            'type_label':type_label.contiguous(),
            'sub_label':sub_label.long().contiguous(),
            'side_label': side_label.long().contiguous(),
            'shape_label': shape_label.long().contiguous(),
            'centroid': torch.tensor(centroid, dtype=torch.float).contiguous(),
            'centroid_center':centroid_features


        }

    def parse_labels_from_filename(self, filename):
        labels = filename.split('.')[0].split('_')[1:]
        if 'Malign' in labels:
            type_label = torch.tensor([0], dtype=torch.float)
            

        elif 'Benign' in labels:
            type_label = torch.tensor([1], dtype=torch.float)
           




        if 'calc' in labels:
            sub_label = torch.tensor([0], dtype=torch.float)
        elif 'mass' in labels:
            sub_label = torch.tensor([1], dtype=torch.float)
        elif 'hybrid' in labels:
            sub_label = torch.tensor([2], dtype=torch.float)







        return  type_label, sub_label,side_label,shape_label


    @staticmethod
    def encode_text_with_clip(text):
     text_token = clip.tokenize([text]).to(device)
     with torch.no_grad():
        text_feature = normal_clip.encode_text(text_token)[0]  
     return text_feature.to(torch.float32)



if __name__ == "__main__":

    i = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(r'train_filenames_small_f.txt', 'r') as file:
        train_filenames_small = file.read().splitlines()
    centroids_filepath = r"..." 

    med_clip_path = r'...'
    train_dataset = MultitaskDataset(image_folder=r'...', image_folder_224=r'...',mask_folder=r'...', filenames=train_filenames_small,med_clip_path=med_clip_path,centroids_filepath=centroids_filepath)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=False)
    for batch in tqdm(train_loader):
        name = batch['idx']

        name_mask = batch['mask_path']

        img_224_feature = batch['img_clip_feature']    #2,256
        print(img_224_feature.shape)
        print(name_mask)

        imgs = batch['image'].to(device)
        assert imgs.min() >= 0 and imgs.max() <= 1, 'True mask indices should be in [0, 1]'
        true_masks = batch['mask'].to(device)
        assert torch.all((true_masks == 0) | true_masks == 1) , 'Mask contains values other than 0 and 1'

        zhixin = batch['centroid_center'].to(device)



 
        type_label = batch['type_label']
        sub_vector = batch['sub_label']



        i+=1

        print(true_masks.shape)

        print("sub_label",sub_vector.shape)
