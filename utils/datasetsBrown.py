import numpy as np
import matplotlib
import cv2
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as F
import random
import scipy.ndimage
import preprocess as prep
from PIL import Image


class LEAFTrain(data.Dataset):
    def __init__(self, list_path, scale=True, mirror=True, color_jitter=True, rotate=True):
        self.jitter_transform=transforms.Compose([
            transforms.ColorJitter(brightness=64. / 255, contrast=0.35, saturation=0.25, hue=0.06)
        ])
        self.list_path = list_path
#        print( 'list_path :',list_path )
        self.ignore_label = 0
        self.scale = scale
        self.is_mirror = mirror
        self.is_jitter = color_jitter
        self.is_rotate = rotate
        self.standardize = True
        self.img_ids = [i_id.strip() for i_id in open(list_path)]

        self.files = []
        for name in self.img_ids:
            img_file, label_file = name.split(',')
            img_file = img_file
            img_name = img_file.split('/')[-1]
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": img_name,
            })

    def __len__(self):
        return len(self.files)

    def generate_scale_label(self, image, label):
        f_scale = 0.5 + random.randint(0, 31) / 10.0
        image = cv2.resize(image, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_LINEAR)
        label = cv2.resize(label, None, fx=f_scale, fy=f_scale, interpolation=cv2.INTER_NEAREST)
        return image, label

    def __getitem__(self, index):
        datafiles = self.files[index]

        image = cv2.imread(datafiles["img"]) #BGR
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = image[:,:,:3]
        imMax = np.max(image)
        if imMax <= 1:
            image = image*255

        label = cv2.imread(datafiles["label"])
        if len(label.shape) > 2:
            label = label[:,:,0]
        if np.max(label) > 1:
            label = label / 255

        image = np.array(image, np.uint8)
        label = np.array(label, np.uint8)
        if self.scale:
            image, label = self.generate_scale_label(image, label)

        if self.is_mirror:
            if random.random()<0.5:
                image = np.flip(image,1)
                label = np.flip(label,1)
        if self.is_rotate:
            angle = random.randint(0, 3)*90
            image = scipy.ndimage.rotate(image,angle)
            label = scipy.ndimage.rotate(label,angle)

        image = Image.fromarray(image)
        if self.is_jitter:
            image = self.jitter_transform(image)

        image = np.asarray(image, np.float32)
        if self.standardize:
            image = prep.normalizeImage(image)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        image = np.transpose(image,(2, 0, 1))

        # image is returned as RGB
        return image.copy(), label.copy(), datafiles["name"]


class LEAFTest(data.Dataset):
    def __init__(self, list_path):
        self.list_path = list_path
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        self.files = []
        for img_file in self.img_ids:
            img_name = img_file.split('/')[-1]
            self.files.append({
                "img": img_file,
                "name": img_name
            })

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"]) #BGR
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = np.asarray(image, np.float32)

        image = np.transpose(image,(2, 0, 1))
        return image.copy(), datafiles["name"]


if __name__ == '__main__':
    pass
