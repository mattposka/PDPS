import numpy as np
import matplotlib
#matplotlib.use('Qt5Agg')
import cv2
from torch.utils import data
import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as F
import random
import matplotlib.pyplot as plt
import scipy.ndimage


class LEAFTrain(data.Dataset):
    def __init__(self, list_path, crop_size=(321, 321), mean=(128, 128, 128),std=(5,5,5), scale=True,
                 mirror=True, color_jitter=True, rotate=True):
        self.jitter_transform=transforms.Compose([
            #transforms.ColorJitter(brightness=64. / 255, contrast=0.25, saturation=0.25, hue=0.04)
            transforms.ColorJitter(brightness=64. / 255, contrast=0.35, saturation=0.25, hue=0.06)
        ])
        self.list_path = list_path
#        print( 'list_path :',list_path )
        self.ignore_label = 0
        self.crop_h, self.crop_w = crop_size
        self.scale = scale
        self.mean = mean
        self.is_mirror = mirror
        self.is_jitter = color_jitter
        self.is_rotate = rotate
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
#        print( 'self.img_ids :',self.img_ids )
        self.files = []
        for name in self.img_ids:
            #TODO fix this later
            #img_file, label_file = name.split(' ')
            #img_file, label_file = name.split('png ')
            #img_file = img_file + 'png'
            img_file, label_file = name.split('npy ')
            img_file = img_file + 'npy'
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
        #image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) #BGR
        #image = cv2.imread(datafiles["img"], cv2.IMREAD_UNCHANGED) #BGR
        #label = cv2.imread(datafiles["label"], cv2.IMREAD_GRAYSCALE)
        # image comes in as BGR
        image = np.load(datafiles["img"]) #BGR
        imMax = np.max(image)
        #print('imMax0 :',imMax)
        if imMax <= 1:
            image[:,:,:3] = image[:,:,:3]*255
        #print('imMax1 :',np.max(image))
        label = np.load(datafiles["label"])
        image = np.array(image, np.uint8)
        label = np.array(label, np.uint8)
        if self.scale:
            image, label = self.generate_scale_label(image, label)

        ####################################################################33

        #image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        ir,ic,_ = image.shape
#        print('image.dtype :',image.dtype)

        #print('imMax2 :',np.max(image))
        img3_bgr = image[:,:,:3]
        #Now image is RGB
        img3_rgb = cv2.cvtColor(img3_bgr,cv2.COLOR_BGR2RGB)
        image3 = Image.fromarray(img3_rgb)

        imageC = image[:,:,3]
        #imageC = imageC.reshape((ir,ic,1))
        label = Image.fromarray(label)
        if self.is_jitter:
            image3 = self.jitter_transform(image3)

        image3 = np.asarray(image3)
        imageC = np.asarray(imageC)
        #image = np.concatenate([image3,imageC],axis=-1)
        image[:,:,:3] = image3
        image[:,:,3] = imageC
        #print('imMax4 :',np.max(image))
        #image = Image.fromarray(image,mode='RGBA')

        if self.is_mirror:
            if random.random()<0.5:
                #image = F.hflip(image)
                #label = F.hflip(label)
                image = np.flip(image,1)
                label = np.flip(label,1)
        if self.is_rotate:
            angle = random.randint(0, 3)*90
            #image = F.rotate(image, angle)
            #label = F.rotate(label, angle)
            image = scipy.ndimage.rotate(image,angle)
            label = scipy.ndimage.rotate(label,angle)

        label = np.asarray(label)

        # image is BGR again
        image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGBA2BGRA)
        #print('imMax5 :',np.max(image))


        #im3c = image[:,:,:3]
        #im4c = image[:,:,3]
        #im3c[im4c==1] = [255,0,0]
        #savename = datafiles['img'].split('/')[-1]
        #savename = savename.replace('C','')
        #savename = savename.replace('Full.npy','Processed.png')
        #cv2.imwrite(savename,im3c)


        image = np.asarray(image, np.float32)

        if np.max(image) > 1:
            image[:,:,:3] = image[:,:,:3]/255.
        #print('imMax6 :',np.max(image))

#####################################################################################3
        # Shouldn't be needed because a blank channel was added to every image in preprocessing
        rows,cols,chans = image.shape
        if chans == 3:
            cir_chan = np.zeros((rows,cols,1))
            image = np.concatenate([image,cir_chan],axis=-1)
#######################################################################################

        image = np.asarray(image, np.float32)
        #print('imMax7 :',np.max(image))
        label = np.asarray(label, np.float32)

        #print('image.max :',np.max(image))
        #print('image.min :',np.min(image))
        image = image.transpose((2, 0, 1))
        return image.copy(), label.copy(), datafiles["name"]


class LEAFTest(data.Dataset):
    def __init__(self, list_path, crop_size=(321, 321), mean=(128, 128, 128), random_crop=False):
        self.list_path = list_path
        self.random_crop = random_crop
        self.crop_h, self.crop_w = crop_size
        self.mean = mean
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
        #image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR) #BGR
        image = cv2.imread(datafiles["img"], cv2.IMREAD_UNCHANGED) #BGR
        image = np.asarray(image, np.float32)
#        image -= self.mean
        img_h, img_w = image.shape[0:2]

        if self.random_crop and (img_h, img_w)!=(self.crop_h, self.crop_w):
            h_off = random.randint(0, img_h - self.crop_h)
            w_off = random.randint(0, img_w - self.crop_w)
            image = np.asarray(image[h_off: h_off + self.crop_h, w_off: w_off + self.crop_w], np.float32)

        image = image.transpose((2, 0, 1))
        return image.copy(), datafiles["name"]


if __name__ == '__main__':
    pass
