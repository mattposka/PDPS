# make train/test dataset
import os
import random
test_imgs_list1 = ['rapa_cam_09_2017-11-22_0400_original.png', 'rapa_cam_09_2017-11-21_2300_original.png',
                  'rapa_cam_13_2017-11-23_0500_original.png', 'rapa_cam_13_2017-11-21_1300_original.png',
                  'rapa_cam_13_2017-11-22_0900_original.png', 'rapa_cam_09_2017-11-23_0500_original.png']
test_imgs_list2 = ['rapa_cam_04_2019-08-04_0800original.png', 'rapa_cam_30_2019-08-09_1600original.png',
                   'rapa_cam_01_2019-08-03_1600original.png', 'rapa_cam_11_2019-11-17_1500original.png',
                   'rapa_cam_33_2019-08-11_0500original.png', 'rapa_cam_04_2019-08-04_1400original.png',
                   'rapa_cam_12_2019-11-16_1600original.png', 'rapa_cam_02_2019-07-04_1700original.png',
                   'rapa_cam_33_2019-08-10_0400original.png', 'rapa_cam_12_2019-11-16_2200original.png',
                   'rapa_cam_13_2019-11-17_0600original.png', 'rapa_cam_01_2019-07-04_1300original.png',
                   'rapa_cam_02_2019-12-08_0000original.png', 'rapa_cam_30_2019-05-24_0600original.png',
                   'rapa_cam_02_2019-08-02_2200original.png', 'rapa_cam_03_2019-08-02_0800original.png',
                   'rapa_cam_02_2019-08-02_0000original.png', 'rapa_cam_32_2019-08-11_0300original.png',
                   'rapa_cam_05_2019-08-02_1300original.png', 'rapa_cam_09_2019-12-07_2100original.png',
                   'rapa_cam_02_2019-07-04_0300original.png', 'rapa_cam_01_2019-08-02_0800original.png',
                   'rapa_cam_08_2019-11-16_1900original.png', 'rapa_cam_31_2019-08-12_0400original.png',
                   'rapa_cam_01_2019-08-01_2200original.png', 'rapa_cam_33_2019-08-09_0800original.png',
                   'rapa_cam_03_2019-08-03_0900original.png', 'rapa_cam_33_2019-08-09_2200original.png',
                   'rapa_cam_29_2019-05-26_0100original.png', 'rapa_cam_04_2019-12-07_1400original.png',
                   'rapa_cam_32_2019-08-10_1700original.png', 'rapa_cam_04_2019-08-03_1100original.png',
                   'rapa_cam_11_2019-11-16_2000original.png', 'rapa_cam_27_2019-10-26_1600original.png',
                   'rapa_cam_29_2019-05-23_1900original.png', 'rapa_cam_01_2019-08-03_0900original.png',
                   'rapa_cam_06_2019-08-02_0900original.png', 'rapa_cam_01_2019-07-04_1800original.png',
                   'rapa_cam_04_2019-08-02_0500original.png', 'rapa_cam_31_2019-08-11_0200original.png',
                   'rapa_cam_30_2019-05-23_1500original.png', 'rapa_cam_32_2019-08-09_1500original.png',
                   'rapa_cam_02_2019-08-03_0300original.png', 'rapa_cam_02_2019-08-03_1300original.png',
                   'rapa_cam_02_2019-08-02_0600original.png', 'rapa_cam_05_2019-08-04_1900original.png',
                   'rapa_cam_30_2019-05-24_0100original.png', 'rapa_cam_13_2019-11-16_2200original.png',
                   'rapa_cam_06_2019-08-02_0500original.png', 'rapa_cam_03_2019-08-02_1800original.png',
                   'rapa_cam_02_2019-08-02_1700original.png', 'rapa_cam_05_2019-08-02_1900original.png',
                   'rapa_cam_04_2019-08-03_0600original.png', 'rapa_cam_02_2019-08-03_1800original.png',
                   'rapa_cam_02_2019-08-02_1200original.png', 'rapa_cam_31_2019-08-09_0600original.png',
                   'rapa_cam_28_2019-10-26_0900original.png', 'rapa_cam_28_2019-10-26_0400original.png',
                   'rapa_cam_06_2019-08-02_1900original.png', 'rapa_cam_05_2019-08-02_0800original.png',
                   'rapa_cam_27_2019-10-27_0100original.png', 'rapa_cam_32_2019-08-11_1000original.png',
                   'rapa_cam_04_2019-08-03_1600original.png', 'rapa_cam_31_2019-08-10_1600original.png',
                   'rapa_cam_31_2019-08-09_0100original.png', 'rapa_cam_05_2019-08-04_0800original.png',
                   'rapa_cam_01_2019-08-02_1800original.png', 'rapa_cam_27_2019-10-26_0600original.png',
                   'rapa_cam_14_2019-11-16_2000original.png', 'rapa_cam_32_2019-08-09_0000original.png',
                   'rapa_cam_05_2019-08-03_1600original.png', 'rapa_cam_01_2019-07-04_2300original.png',
                   'rapa_cam_29_2019-05-26_0600original.png', 'rapa_cam_05_2019-08-03_2100original.png',
                   'rapa_cam_06_2019-08-03_0000original.png', 'rapa_cam_32_2019-08-10_1100original.png',
                   'rapa_cam_32_2019-08-11_1900original.png', 'rapa_cam_05_2019-08-04_1300original.png',
                   'rapa_cam_03_2019-12-07_0900original.png', 'rapa_cam_01_2019-08-03_0400original.png',
                   'rapa_cam_29_2019-05-26_1100original.png', 'rapa_cam_04_2019-08-03_2200original.png',
                   'rapa_cam_05_2019-08-03_1100original.png', 'rapa_cam_33_2019-08-09_0200original.png',
                   'rapa_cam_03_2019-08-02_0300original.png']
test_imgs_list = test_imgs_list1 + test_imgs_list2

# make training dataset
def mk_train():
    version = "feb2021"
    root_dir = "/data/leaf_train/green"
    root_dir = os.path.join(root_dir, version)
    TumorTxt = root_dir + '/256_train_stride_128/green.txt'
    NormalTxt = root_dir + '/256_train_stride_128/normal.txt'
    TrainTxt = root_dir + '/256_train_stride_128/train.txt'

    TumorFile = open(TumorTxt, 'r')
    NormalFile = open(NormalTxt, 'r')
    TrainFile = open(TrainTxt, 'w')

    TumorLines = TumorFile.readlines()
    n_samples = int(len(TumorLines) / 3)
    NormalLines = NormalFile.readlines()
    random.shuffle(NormalLines)
    NormalLines = NormalLines[:n_samples]

    for line in TumorLines:
        TrainFile.write(line)
    for line in NormalLines:
        TrainFile.write(line)
    TrainFile.close()
    TrainFile = open(TrainTxt, 'r')
    print(len(TrainFile.readlines()))
    # 1002 brown
    # 1142 green
    # 3686 green
    # 13862 green


# make testing dataset
def mk_test():
    version = "feb2021"
    root_dir = "/data/leaf_train"
    root_dir = os.path.join(root_dir, version)
    TestPath = root_dir + '/images'
    TxtPath = root_dir + '/512_s64'
    if not os.path.exists(TxtPath):
        os.makedirs(TxtPath)
    SlideList = os.listdir(TestPath)
    for Slide in SlideList:
        fname = Slide + '.txt'
        ftxt = os.path.join(TxtPath, fname)
        f = open(ftxt, 'w')
        SlideDir = os.path.join(TestPath, Slide)
        PatchList = os.listdir(SlideDir)
        for PatchName in PatchList:
            PatchFile = os.path.join(SlideDir, PatchName)
            f.write(PatchFile + '\n')


if __name__ == "__main__":
    mk_train()
