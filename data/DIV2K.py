import torch.utils.data as data
import os.path
import cv2
import numpy as np
from data import common
import random

def default_loader(path):
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0]]

def npy_loader(path):
    return np.load(path)

IMG_EXTENSIONS = [
    '.png', '.npy',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

class div2k(data.Dataset):
    """Dataset for DIV2K, i.e., train dataset.
    """
    def __init__(self, opt):
        self.opt = opt
        self.scale = self.opt.scale
        self.root = self.opt.root
        self.ext = self.opt.ext   # '.png' or '.npy'(default)
        self.train = True if self.opt.phase == 'train' else False
        self._set_filesystem(self.root)
        self.images_hr, self.images_lr = self._scan()

    def _set_filesystem(self, dir_data):
        self.dir_hr = os.path.join(self.root, 'DIV2K_train_HR')
        self.dir_lr = os.path.join(self.root, 'DIV2K_train_LR_bicubic/X' + str(self.scale))

    def __getitem__(self, idx):
        lr, hr = self._load_file(idx) #load image (load ndarray instead to accelerate).
        lr, hr = self._get_patch(lr, hr) #data augment
        lr, hr = common.set_channel(lr, hr, n_channels=self.opt.n_colors)
        lr, hr = common.np2Tensor(lr, hr, rgb_range=self.opt.rgb_range)
        return lr, hr

    def __len__(self):
        return self.opt.n_train

    def _get_patch(self, img_in, img_tar):
        patch_size = self.opt.patch_size
        scale = self.scale
        if self.train:
            img_in, img_tar = common.get_patch(
                img_in, img_tar, patch_size=patch_size, scale=scale)
            img_in, img_tar = common.augment(img_in, img_tar)
        else:
            ih, iw = img_in.shape[:2]
            img_tar = img_tar[0:ih * scale, 0:iw * scale, :]
        return img_in, img_tar

    def _scan(self):
        list_hr = sorted(make_dataset(self.dir_hr))
        list_lr = sorted(make_dataset(self.dir_lr))
        return list_hr, list_lr

    def _load_file(self, idx):
        if self.ext == '.npy':
            lr = npy_loader(self.images_lr[idx])
            hr = npy_loader(self.images_hr[idx])
        else:
            lr = default_loader(self.images_lr[idx])
            hr = default_loader(self.images_hr[idx])
        return lr, hr

class Flickr(data.Dataset):
    """Dataset for Flickr, i.e., train dataset.
    """
    def __init__(self, opt,root, scale=4,ext='.png'):
        self.opt=opt
        self.scale = scale
        self.root = root
        self.ext = ext
        self.train = True
        self.images_hr = sorted(make_dataset(self.root))

    def __getitem__(self, idx):
        hr = self._load_file(idx) #load image (load ndarray instead to accelerate).
        hr = self._get_patch(hr) #data augment
        hr = common.set_channel(hr, n_channels=self.opt.n_colors)[0].copy()
        lr = common.imresize_np(hr, 1/self.scale,True)
        lr, hr = common.np2Tensor(lr, hr, rgb_range=self.opt.rgb_range)
        return lr, hr

    def __len__(self):
        return len(self.images_hr)


    def _get_patch(self, img_tar):
        patch_size = self.opt.patch_size
        h, w = img_tar.shape[:2]
        if self.train:
            ix = random.randrange(0, w - patch_size + 1)
            iy = random.randrange(0, h - patch_size + 1)
            img_tar = img_tar[iy:iy+patch_size,ix:ix+patch_size,:]
            img_tar = common.augment(img_tar)[0]
        return img_tar

    def _load_file(self, idx):
        if self.ext == '.npy':
            hr = npy_loader(self.images_hr[idx])
        else:
            hr = default_loader(self.images_hr[idx])
        return  hr

class RepeatDataset(data.Dataset):
    def __init__(self, dataset, repeat=20):
        self.dataset = dataset
        self.repeat = repeat
        self.len = len(dataset)

    def __getitem__(self, idx):
        idx = idx % self.len
        return self.dataset.__getitem__(idx)

    def __len__(self):
        return self.len*self.repeat
