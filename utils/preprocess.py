# this file includes several pre-processing functions.
import os
from scipy.misc import imread, imsave
import random
import numpy as np
import cv2
from shutil import copyfile
test_imgs_list = ['rapa_cam_09_2017-11-22_0400_original.png', 'rapa_cam_09_2017-11-21_2300_original.png',
                  'rapa_cam_13_2017-11-23_0500_original.png', 'rapa_cam_13_2017-11-21_1300_original.png',
                  'rapa_cam_13_2017-11-22_0900_original.png', 'rapa_cam_09_2017-11-23_0500_original.png']

# rename the image for better coding
def renamefile():
    file_pth = "/data/AutoPheno"
    files = os.listdir(file_pth)
    files.sort()
    for file_name in files:
        img_pth = os.path.join(file_pth, file_name)
        new_file_name = file_name.replace(" ", "_")
        print(new_file_name)
        new_img_pth = os.path.join(file_pth, new_file_name)
        os.rename(img_pth, new_img_pth)


def renamefile2():
    file_pth = "/data/AutoPheno/imgs"
    files = os.listdir(file_pth)
    files.sort()
    for file_name in files:
        if not "rapa_cam_9" in file_name:
            continue
        img_pth = os.path.join(file_pth, file_name)
        new_file_name = file_name.replace("rapa_cam_9", "rapa_cam_09")
        print(new_file_name)
        new_img_pth = os.path.join(file_pth, new_file_name)
        os.rename(img_pth, new_img_pth)


def split_train_test():
    # train/test split
    file_pth = "/data/AutoPheno/new_imgs_200527/lesion_training_set_2"
    files = os.listdir(file_pth)
    files.sort()
    imgs = []
    for img_name in files:
        if not "original" in img_name:
            continue
        imgs.append(img_name)
    imgs.sort()
    print(len(imgs))
    random.shuffle(imgs)
    print("train set:", imgs[:75])  # 91 in total
    print("test set:", imgs[75:])
    # train
    # set: ['rapa_cam_09_2017-11-21_0800_original.png', 'rapa_cam_13_2017-11-22_1400_original.png',
    #       'rapa_cam_09_2017-11-21_0300_original.png', 'rapa_cam_09_2017-11-22_0900_original.png',
    #       'rapa_cam_09_2017-11-22_1400_original.png', 'rapa_cam_13_2017-11-21_2300_original.png',
    #       'rapa_cam_13_2017-11-21_0300_original.png', 'rapa_cam_13_2017-11-21_0800_original.png',
    #       'rapa_cam_09_2017-11-21_1300_original.png', 'rapa_cam_13_2017-11-21_1800_original.png',
    #       'rapa_cam_09_2017-11-21_1800_original.png', 'rapa_cam_13_2017-11-22_0400_original.png',
    #       'rapa_cam_09_2017-11-22_1900_original.png', 'rapa_cam_09_2017-11-23_0000_original.png',
    #       'rapa_cam_13_2017-11-22_1900_original.png', 'rapa_cam_13_2017-11-23_0000_original.png']
    # test
    # set: ['rapa_cam_09_2017-11-22_0400_original.png', 'rapa_cam_09_2017-11-21_2300_original.png',
    #       'rapa_cam_13_2017-11-23_0500_original.png', 'rapa_cam_13_2017-11-21_1300_original.png',
    #       'rapa_cam_13_2017-11-22_0900_original.png', 'rapa_cam_09_2017-11-23_0500_original.png']
    # new set
    # train
    # set: ['rapa_cam_05_2019-08-01_2200original.png', 'rapa_cam_29_2019-05-24_0700original.png',
    #       'rapa_cam_08_2019-11-17_2100original.png', 'rapa_cam_01_2019-08-02_1300original.png',
    #       'rapa_cam_33_2019-08-11_0000original.png', 'rapa_cam_05_2019-08-03_0500original.png',
    #       'rapa_cam_02_2019-07-04_0700original.png', 'rapa_cam_32_2019-08-09_1000original.png',
    #       'rapa_cam_29_2019-05-25_1100original.png', 'rapa_cam_04_2019-08-04_1900original.png',
    #       'rapa_cam_03_2019-08-02_1300original.png', 'rapa_cam_32_2019-08-10_0100original.png',
    #       'rapa_cam_32_2019-08-10_0600original.png', 'rapa_cam_32_2019-08-09_0500original.png',
    #       'rapa_cam_30_2019-08-09_0600original.png', 'rapa_cam_31_2019-08-09_1100original.png',
    #       'rapa_cam_04_2019-08-02_0000original.png', 'rapa_cam_11_2019-11-17_0600original.png',
    #       'rapa_cam_29_2019-05-26_1600original.png', 'rapa_cam_04_2019-08-02_1500original.png',
    #       'rapa_cam_06_2019-08-02_0000original.png', 'rapa_cam_11_2019-12-08_0100original.png',
    #       'rapa_cam_27_2019-10-26_1200original.png', 'rapa_cam_01_2019-08-02_2300original.png',
    #       'rapa_cam_04_2019-08-02_2000original.png', 'rapa_cam_32_2019-08-10_2200original.png',
    #       'rapa_cam_33_2019-08-09_1400original.png', 'rapa_cam_11_2019-11-17_2100original.png',
    #       'rapa_cam_31_2019-08-11_2200original.png', 'rapa_cam_02_2019-07-04_1200original.png',
    #       'rapa_cam_33_2019-08-10_1800original.png', 'rapa_cam_04_2019-08-04_0300original.png',
    #       'rapa_cam_06_2019-08-03_0500original.png', 'rapa_cam_31_2019-08-11_1700original.png',
    #       'rapa_cam_03_2019-08-03_0400original.png', 'rapa_cam_01_2019-12-07_1100original.png',
    #       'rapa_cam_29_2019-05-24_0100original.png', 'rapa_cam_03_2019-08-03_1400original.png',
    #       'rapa_cam_12_2019-11-17_1000original.png', 'rapa_cam_30_2019-08-09_2100original.png',
    #       'rapa_cam_01_2019-08-02_0300original.png', 'rapa_cam_12_2019-11-17_1500original.png',
    #       'rapa_cam_12_2019-11-17_0200original.png', 'rapa_cam_30_2019-08-09_1100original.png',
    #       'rapa_cam_10_2019-12-07_1500original.png', 'rapa_cam_31_2019-08-10_2100original.png',
    #       'rapa_cam_06_2019-08-02_1400original.png', 'rapa_cam_13_2019-11-17_2000original.png',
    #       'rapa_cam_29_2019-05-25_1500original.png', 'rapa_cam_29_2019-05-25_2100original.png',
    #       'rapa_cam_13_2019-11-17_1400original.png', 'rapa_cam_02_2019-08-03_0800original.png',
    #       'rapa_cam_08_2019-11-17_0600original.png', 'rapa_cam_04_2019-08-03_0100original.png',
    #       'rapa_cam_13_2019-11-18_0200original.png', 'rapa_cam_30_2019-08-09_0100original.png',
    #       'rapa_cam_12_2019-11-17_1400original.png', 'rapa_cam_30_2019-05-23_2000original.png',
    #       'rapa_cam_31_2019-08-12_0900original.png', 'rapa_cam_05_2019-08-04_0300original.png',
    #       'rapa_cam_32_2019-08-09_2000original.png', 'rapa_cam_08_2019-12-07_1800original.png',
    #       'rapa_cam_33_2019-08-10_1000original.png', 'rapa_cam_01_2019-07-04_0300original.png',
    #       'rapa_cam_05_2019-08-03_0000original.png', 'rapa_cam_31_2019-08-11_1200original.png',
    #       'rapa_cam_28_2019-10-26_1900original.png', 'rapa_cam_01_2019-07-04_0800original.png',
    #       'rapa_cam_31_2019-08-11_0700original.png', 'rapa_cam_03_2019-08-02_2300original.png',
    #       'rapa_cam_10_2019-12-07_0700original.png', 'rapa_cam_05_2019-08-02_0300original.png',
    #       'rapa_cam_29_2019-05-26_2200original.png', 'rapa_cam_13_2019-11-16_1700original.png',
    #       'rapa_cam_04_2019-08-02_1000original.png']
    # test
    # set: ['rapa_cam_04_2019-08-04_0800original.png', 'rapa_cam_30_2019-08-09_1600original.png',
    #       'rapa_cam_01_2019-08-03_1600original.png', 'rapa_cam_11_2019-11-17_1500original.png',
    #       'rapa_cam_33_2019-08-11_0500original.png', 'rapa_cam_04_2019-08-04_1400original.png',
    #       'rapa_cam_12_2019-11-16_1600original.png', 'rapa_cam_02_2019-07-04_1700original.png',
    #       'rapa_cam_33_2019-08-10_0400original.png', 'rapa_cam_12_2019-11-16_2200original.png',
    #       'rapa_cam_13_2019-11-17_0600original.png', 'rapa_cam_01_2019-07-04_1300original.png',
    #       'rapa_cam_02_2019-12-08_0000original.png', 'rapa_cam_30_2019-05-24_0600original.png',
    #       'rapa_cam_02_2019-08-02_2200original.png', 'rapa_cam_03_2019-08-02_0800original.png',
    #       'rapa_cam_02_2019-08-02_0000original.png', 'rapa_cam_32_2019-08-11_0300original.png',
    #       'rapa_cam_05_2019-08-02_1300original.png', 'rapa_cam_09_2019-12-07_2100original.png',
    #       'rapa_cam_02_2019-07-04_0300original.png', 'rapa_cam_01_2019-08-02_0800original.png',
    #       'rapa_cam_08_2019-11-16_1900original.png', 'rapa_cam_31_2019-08-12_0400original.png',
    #       'rapa_cam_01_2019-08-01_2200original.png', 'rapa_cam_33_2019-08-09_0800original.png',
    #       'rapa_cam_03_2019-08-03_0900original.png', 'rapa_cam_33_2019-08-09_2200original.png',
    #       'rapa_cam_29_2019-05-26_0100original.png', 'rapa_cam_04_2019-12-07_1400original.png',
    #       'rapa_cam_32_2019-08-10_1700original.png', 'rapa_cam_04_2019-08-03_1100original.png',
    #       'rapa_cam_11_2019-11-16_2000original.png', 'rapa_cam_27_2019-10-26_1600original.png',
    #       'rapa_cam_29_2019-05-23_1900original.png', 'rapa_cam_01_2019-08-03_0900original.png',
    #       'rapa_cam_06_2019-08-02_0900original.png', 'rapa_cam_01_2019-07-04_1800original.png',
    #       'rapa_cam_04_2019-08-02_0500original.png', 'rapa_cam_31_2019-08-11_0200original.png',
    #       'rapa_cam_30_2019-05-23_1500original.png', 'rapa_cam_32_2019-08-09_1500original.png',
    #       'rapa_cam_02_2019-08-03_0300original.png', 'rapa_cam_02_2019-08-03_1300original.png',
    #       'rapa_cam_02_2019-08-02_0600original.png', 'rapa_cam_05_2019-08-04_1900original.png',
    #       'rapa_cam_30_2019-05-24_0100original.png', 'rapa_cam_13_2019-11-16_2200original.png',
    #       'rapa_cam_06_2019-08-02_0500original.png', 'rapa_cam_03_2019-08-02_1800original.png',
    #       'rapa_cam_02_2019-08-02_1700original.png', 'rapa_cam_05_2019-08-02_1900original.png',
    #       'rapa_cam_04_2019-08-03_0600original.png', 'rapa_cam_02_2019-08-03_1800original.png',
    #       'rapa_cam_02_2019-08-02_1200original.png', 'rapa_cam_31_2019-08-09_0600original.png',
    #       'rapa_cam_28_2019-10-26_0900original.png', 'rapa_cam_28_2019-10-26_0400original.png',
    #       'rapa_cam_06_2019-08-02_1900original.png', 'rapa_cam_05_2019-08-02_0800original.png',
    #       'rapa_cam_27_2019-10-27_0100original.png', 'rapa_cam_32_2019-08-11_1000original.png',
    #       'rapa_cam_04_2019-08-03_1600original.png', 'rapa_cam_31_2019-08-10_1600original.png',
    #       'rapa_cam_31_2019-08-09_0100original.png', 'rapa_cam_05_2019-08-04_0800original.png',
    #       'rapa_cam_01_2019-08-02_1800original.png', 'rapa_cam_27_2019-10-26_0600original.png',
    #       'rapa_cam_14_2019-11-16_2000original.png', 'rapa_cam_32_2019-08-09_0000original.png',
    #       'rapa_cam_05_2019-08-03_1600original.png', 'rapa_cam_01_2019-07-04_2300original.png',
    #       'rapa_cam_29_2019-05-26_0600original.png', 'rapa_cam_05_2019-08-03_2100original.png',
    #       'rapa_cam_06_2019-08-03_0000original.png', 'rapa_cam_32_2019-08-10_1100original.png',
    #       'rapa_cam_32_2019-08-11_1900original.png', 'rapa_cam_05_2019-08-04_1300original.png',
    #       'rapa_cam_03_2019-12-07_0900original.png', 'rapa_cam_01_2019-08-03_0400original.png',
    #       'rapa_cam_29_2019-05-26_1100original.png', 'rapa_cam_04_2019-08-03_2200original.png',
    #       'rapa_cam_05_2019-08-03_1100original.png', 'rapa_cam_33_2019-08-09_0200original.png',
    #       'rapa_cam_03_2019-08-02_0300original.png']


# calculate the average of B, G, R channels
def cal_BGR_mean():
    file_pth = "/data/AutoPheno/imgs"
    files = os.listdir(file_pth)
    files.sort()
    tr_imgs_arr = []
    img_size = (2048, 2048)
    for img_name in files:
        if not "original" in img_name:
            continue
        if img_name in test_imgs_list:
            continue
        img_pth = os.path.join(file_pth, img_name)
        img_arr = cv2.imread(img_pth, cv2.IMREAD_COLOR)  # BGR
        img_arr = cv2.resize(img_arr, img_size, interpolation=cv2.INTER_LINEAR)
        tr_imgs_arr.append(img_arr)
    tr_imgs_arr = np.array(tr_imgs_arr)
    print("BGR:", [tr_imgs_arr[:, :, :, 0].mean(), tr_imgs_arr[:, :, :, 1].mean(), tr_imgs_arr[:, :, :, 2].mean()])


# calculate the class distribution for brown disease.
def cal_brown_cls_distri():
    file_pth = "/data/AutoPheno/imgs"
    files = os.listdir(file_pth)
    files.sort()
    num_brown = 0
    num_all = 0
    img_size = (2048, 2048)
    for img_name in files:
        if not "brown" in img_name:
            continue
        if img_name in test_imgs_list:
            continue
        lbl_pth = os.path.join(file_pth, img_name)
        print(lbl_pth)
        lbl = cv2.imread(lbl_pth, cv2.IMREAD_COLOR)
        lbl = cv2.resize(lbl, img_size, cv2.INTER_LINEAR)
        num_brown += np.sum(np.all(lbl==[0, 0, 255], axis=2))
        num_all += img_size[0]*img_size[1]
    print("others: brown", num_all-num_brown, num_brown)


def resave():
    # resave image in a correct form
    img_dir_pth = "/workspace/data/AutoPheno/imgs"
    img_dir = os.listdir(img_dir_pth)
    img_dir.sort()
    for img_name in img_dir:
        img_pth = os.path.join(img_dir_pth, img_name)
        print(img_pth)
        img = imread(img_pth)
        img = img[:, :, :3]
        imsave(img_pth, img)

def random_save(n_test=100):
    # randomly choose some images for testing
    random_seed = 1234
    random.seed(random_seed)
    img_dir_pth = "/data/AutoPheno/images.2020-02-28_1300"
    img_list = os.listdir(img_dir_pth)
    img_list.sort()
    random.shuffle(img_list)
    selected_img_list = img_list[:n_test]

    random_img_dir_pth = "/data/AutoPheno/random_test_imgs"
    for img_name in selected_img_list:
        img_file = os.path.join(random_img_dir_pth, img_name)
        src_file = os.path.join(img_dir_pth, img_name)
        copyfile(src_file, img_file)
        print("copy from", src_file, "to", img_file)

if __name__ == "__main__":
    # renamefile()
    # renamefile2()
    # resave()
    # random_save()
    pass
