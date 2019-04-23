import numpy as np
from PIL import Image, ImageFilter
import pandas as pd
import os
import torch as t
from imageio import imread
from torch.utils.data import TensorDataset
from os import path

img_size = 32

class Augmenter:
    def __init__(self, hflip=True, rotate=True, blurring=False):
        self.hflip = hflip
        self.rotate = rotate
        self.blurring = blurring

    def extract_mutiview(self, voxel):
        x, y, z = voxel.shape
        return [voxel[x // 2, :, :],
                voxel[:, y // 2, :],
                voxel[:, :, z // 2]]

    def augment(self, voxel):
        for view in self.extract_mutiview(voxel):
            im = Image.fromarray(view, mode='L')
            yield im
            if self.hflip:
                yield im.transpose(Image.FLIP_LEFT_RIGHT)
            if self.rotate:
                yield im.transpose(Image.ROTATE_90)
                yield im.transpose(Image.ROTATE_180)
                yield im.transpose(Image.ROTATE_270)
            if self.blurring:
                yield im.filter(ImageFilter.GaussianBlur(1))


def resize(im):
    im = im.crop((8, 8, 40, 40))
    return im


def generate_dataset(dir):
    df = pd.read_csv(dir+'/labels.csv')
    df['testing'] = 1
    voxels = np.zeros((len(df),48,48,48), dtype=np.uint8)
    augmenter = Augmenter(hflip=True, rotate=True, blurring=True)
    augmenter2 = Augmenter(hflip=False, rotate=False, blurring=False)
    for i, row in df.iterrows():
        voxels[int(row.id)] = np.load('{0}/{1:.0f}.npy'.format(dir,row.id))

    for i in range(10):
        folder = '{0}/{1}/'.format(dir,i)
        if not os.path.exists(folder):
            os.makedirs(folder)
        tests = df[df.fold == i].copy()
        trains = df[df.fold != i].copy()
        trains.testing = 0
        new_df = pd.concat([tests, trains])
        new_df.to_csv(folder+'/labels.csv', index=False)

        for j, row in tests.iterrows():
            voxel = voxels[int(row.id)]
            for e,im in enumerate(augmenter2.augment(voxel)):
                im2 = resize(im)
                im2.save('{0}{1:.0f}.{2}.png'.format(folder, row.id, e))

        for j, row in trains.iterrows():
            voxel = voxels[int(row.id)]
            for e,im in enumerate(augmenter.augment(voxel)):
                im2 = resize(im)
                im2.save('{0}{1:.0f}.{2}.png'.format(folder, row.id, e))

def get_dataset(dir):
    df = pd.read_csv(path.join(dir, 'labels.csv'))
    df_test = df[df.testing==1]
    df_train = df[df.testing == 0]

    num_data = len(df_train)
    aug_size = 18
    x = t.zeros((num_data * aug_size, 1, img_size, img_size))
    y = t.zeros((num_data * aug_size, 1))
    c = 0
    for i, row in df_train.iterrows():
        id = int(row.id)
        for j in range(aug_size):
            im = imread(path.join(dir,f'{id:.0f}.{j}.png'))
            x[c * aug_size + j, 0, :, :] = t.from_numpy(im)
            y[c * aug_size + j][0] = row.malignancy_th
        c += 1

    mu = x.mean()
    sd = x.std()
    x = (x - mu) / sd

    trainset = TensorDataset(x, y)
    aug_size = 3
    num_data = len(df_test)
    x = t.zeros((num_data*aug_size, 1, img_size, img_size))
    y = t.zeros((num_data*aug_size, 1))
    c = 0
    for i, row in df_test.iterrows():
        id = int(row.id)
        for j in range(aug_size):
            im = imread(path.join(dir, f'{id:.0f}.{j}.png'))
            x[c * aug_size + j, 0, :, :] = t.from_numpy(im)
            y[c * aug_size + j][0] = row.malignancy_th
        c += 1

    x = (x - mu) / sd
    testset = TensorDataset(x, y)

    return trainset, testset

def get_dataset3d(dir):
    df = pd.read_csv(path.join(dir, 'labels.csv'))
    df_test = df[df.testing==1]
    df_train = df[df.testing == 0]

    num_data = len(df_train)
    aug_size = 18
    x = t.zeros((num_data * aug_size, 3, img_size, img_size))
    y = t.zeros((num_data * aug_size, 1))
    c = 0
    for i, row in df_train.iterrows():
        id = int(row.id)
        for j in range(aug_size):
            im = imread(path.join(dir, f'{id:.0f}.{j}.png'))
            x[c * aug_size + j, 0, :, :] = t.from_numpy(im)
            y[c * aug_size + j][0] = row.malignancy_th
            x[c * aug_size + j, 1, :, :] = x[c * aug_size + j, 0, :, :]
            x[c * aug_size + j, 2, :, :] = x[c * aug_size + j, 0, :, :]
        c += 1

    mu = x.mean()
    sd = x.std()
    x = (x - mu) / sd
    trainset = TensorDataset(x, y)
    aug_size = 3
    num_data = len(df_test)
    x = t.zeros((num_data*aug_size, 3, img_size, img_size))
    y = t.zeros((num_data*aug_size, 1))
    c = 0
    for i, row in df_test.iterrows():
        id = int(row.id)
        for j in range(aug_size):
            im = imread(path.join(dir, f'{id:.0f}.{j}.png'))
            x[c * aug_size + j, 0, :, :] = t.from_numpy(im)
            y[c * aug_size + j][0] = row.malignancy_th
            x[c * aug_size + j, 1, :, :] = x[c * aug_size + j, 0, :, :]
            x[c * aug_size + j, 2, :, :] = x[c * aug_size + j, 0, :, :]
        c += 1

    x = (x - mu) / sd
    testset = TensorDataset(x, y)

    return trainset, testset


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 2:
        generate_dataset(sys.argv[1])
    else:
        print("run \"python3 preprocessing.py <path to output directory>\"")