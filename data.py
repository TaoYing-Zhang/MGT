import os
import cv2
from torch.utils import data
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import torch
import random

to_tensor = transforms.Compose([transforms.ToTensor()])
EPSILON = 1e-5


class TestDataset(data.Dataset):
    def __init__(self, data_dir, Vis_RGB=False, transform=to_tensor):
        super().__init__()
        self.RGB = Vis_RGB
        dirname = os.listdir(data_dir)  # subdirectory of the dataset
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'Inf':
                self.inf_path = temp_path  # inf path
            else:
                self.vis_path = temp_path  # vis path

        self.name_list = os.listdir(self.inf_path)  # the name of the images in the subdirectory
        self.transform = transform

    def __getitem__(self, index):
        name = self.name_list[index]  # the name of the current image

        inf_image = cv2.imread(os.path.join(self.inf_path, name), 0)  # infrared images
        # inf_image = cv2.resize(inf_image, (689, 509))
        inf_image = self.transform(inf_image)
        if self.RGB:
            vis_image = cv2.imread(os.path.join(self.vis_path, name))
            # y = vis_image
            vis_image = self.transform(vis_image)
            vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
            return vis_y_image, vis_cb_image, vis_cr_image, inf_image, name  # , y
        else:
            vis_image = cv2.imread(os.path.join(self.vis_path, name), 0)
            # vis_image = cv2.resize(vis_image, (689, 509))
            vis_image = self.transform(vis_image)
            return vis_image, inf_image, name

    def __len__(self):
        return len(self.name_list)


class TrainDataset(data.Dataset):
    def __init__(self, data_dir, Vis_RGB=False, crop_flag=False, patch_size=256, transform=to_tensor):
        super().__init__()
        self.RGB = Vis_RGB
        dirname = os.listdir(data_dir)  # subdirectory of the dataset
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'Inf':
                self.inf_path = temp_path  # inf path
            else:
                self.vis_path = temp_path  # vis path

        self.name_list = os.listdir(self.inf_path)  # the name of the images in the subdirectory
        self.transform = transform
        self.crop_flag = crop_flag
        self.patch_size = patch_size

    def __getitem__(self, index):
        name = self.name_list[index]  # the name of the current image

        inf_image = cv2.imread(os.path.join(self.inf_path, name), 0)  # infrared images
        # inf_image = cv2.resize(inf_image, (170, 170))
        inf_image = self.transform(inf_image)
        if self.RGB:
            vis_image = cv2.imread(os.path.join(self.vis_path, name))
            vis_image = self.transform(vis_image)
            # vis_image = cv2.resize(vis_image, (170, 170))
            vis_image, _, _ = RGB2YCrCb(vis_image)
        else:
            vis_image = cv2.imread(os.path.join(self.vis_path, name), 0)
            vis_image = self.transform(vis_image)
        if self.crop_flag:
            _, H, W = vis_image.shape
            rnd_h = random.randint(0, max(0, H - self.patch_size))
            rnd_w = random.randint(0, max(0, W - self.patch_size))
            vis_image = vis_image[:, rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]
            inf_image = inf_image[:, rnd_h:rnd_h + self.patch_size, rnd_w:rnd_w + self.patch_size]

        return vis_image, inf_image, name

    def __len__(self):
        return len(self.name_list)


def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)


def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式

    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """

    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    # Y = clamp(Y)
    # Cr = clamp(Cr)
    # Cb = clamp(Cb)
    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式

    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    # print(ycrcb.shape)
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.reshape(W, H, C)
    out = clamp(out)
    return out


if __name__ == '__main__':
    D = TestDataset('./Dataset/test/MSRS', True)
    dl = DataLoader(D, shuffle=True, batch_size=1)
    print(len(dl))
    for i, data in enumerate(dl):
        vis_image, a, b, inf_image, name, y = data
        # vis_image, inf_image, name = data
        o = YCrCb2RGB(vis_image[0], a[0], b[0])
        print(torch.mean((o*255) - y[0]))
        break
