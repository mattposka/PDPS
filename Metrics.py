# measure the performance of our model
import os
from utils.transforms import im2vl
from skimage.measure import label, regionprops
from skimage.morphology import closing, square, remove_small_objects
from scipy.misc import imread, imresize
import numpy as np

gt_pth = "/data/AutoPheno/imgs_all/"
pd_pth = "/data/AutoPheno/green/200527/PatchNet/npz-fb/whole_fig_post/10700"
#  "/data/AutoPheno/green/PatchNet/npz/whole_fig_post/1800"
#  "/data/AutoPheno/brown/PatchNet/npz/CRF"

pds = os.listdir(pd_pth)
pds.sort()
mDice = []

for pd_name in pds:
    pd = os.path.join(pd_pth, pd_name)
    pd_vl = im2vl(imread(pd))

    pd_close = closing(pd_vl, square(3))
    labeled_pd = label(pd_close, connectivity=2)
    new_labeled_pd = remove_small_objects(labeled_pd, min_size=5000, connectivity=2)
    # remove non-circle region
    props = regionprops(new_labeled_pd)
    for i, reg in enumerate(props):
        if reg.eccentricity > 0.92:
            new_labeled_pd[new_labeled_pd == reg.label] = 0
    new_pd = np.asarray(new_labeled_pd != 0, dtype=np.uint8)
    pd_vl = new_pd

    gt_name = pd_name.replace("original_Map", "green")
    gt = os.path.join(gt_pth, gt_name)
    try:
        gt_vl = im2vl(imread(gt))
    except:
        print(gt_name + " is missing!")
        continue
    gt_vl = imresize(gt_vl, pd_vl.shape)
    overlap = np.sum(np.logical_and(pd_vl, gt_vl))
    union = np.sum(pd_vl) + np.sum(gt_vl)
    res = 2*overlap/union
    print(pd_name, res)
    mDice.append(res)
print("mean Dice", np.mean(mDice))

# brown CRF
# rapa_cam_09_2017-11-21_2300_original_Map.png 0.8162936161408237
# rapa_cam_09_2017-11-22_0400_original_Map.png 0.9344598213246195
# rapa_cam_09_2017-11-23_0500_original_Map.png 0.8152400159243784
# rapa_cam_13_2017-11-21_1300_original_Map.png 0.5045799076611609
# rapa_cam_13_2017-11-22_0900_original_Map.png 0.8958815008106527
# rapa_cam_13_2017-11-23_0500_original_Map.png 0.9140564050098338
# mean Dice 0.8134185444785782

# brown
# rapa_cam_09_2017-11-21_2300_original_Map.png 0.6861177840867729
# rapa_cam_09_2017-11-22_0400_original_Map.png 0.7974052955627673
# rapa_cam_09_2017-11-23_0500_original_Map.png 0.8282056468798146
# rapa_cam_13_2017-11-21_1300_original_Map.png 0.3060353814816597
# rapa_cam_13_2017-11-22_0900_original_Map.png 0.7023604576494369
# rapa_cam_13_2017-11-23_0500_original_Map.png 0.8297601304561918
# mean Dice 0.6916474493527739

# green
# rapa_cam_09_2017-11-21_2300_original_Map.png 0.7614843656879494
# rapa_cam_09_2017-11-22_0400_original_Map.png 0.860228750364886
# rapa_cam_09_2017-11-23_0500_original_Map.png 0.8695406969993531
# rapa_cam_13_2017-11-21_1300_original_Map.png 0.44298245614035087
# rapa_cam_13_2017-11-22_0900_original_Map.png 0.8111305766047803
# rapa_cam_13_2017-11-23_0500_original_Map.png 0.8817172033470765
# mean Dice 0.7711806748573994

# green CRF
# rapa_cam_09_2017-11-21_2300_original_Map.png 0.8615768413098007
# rapa_cam_09_2017-11-22_0400_original_Map.png 0.8740331756310656
# rapa_cam_09_2017-11-23_0500_original_Map.png 0.7987432590100497
# rapa_cam_13_2017-11-21_1300_original_Map.png 0.5005834877859826
# rapa_cam_13_2017-11-22_0900_original_Map.png 0.8861312434064029
# rapa_cam_13_2017-11-23_0500_original_Map.png 0.8757686523630875
# mean Dice 0.7994727765843982

# green
# mean Dice 0.7520124325767369
# green 10000
# mean Dice 0.7723868749277915
# green 5000
# mean Dice 0.7960500916232719
# mean Dice 0.8053091548055505
# green CRF
# mean Dice 0.5854599376889339

