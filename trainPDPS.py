# U-Net for LEAF Dataset
# Frozen BN when set is_training as false
from __future__ import print_function
import argparse
import torch
from torch.utils import data
import torch.nn as nn
import numpy as np
import cv2
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import os
import os.path as osp
from utils.datasetsBrown import LEAFTrain
from model.u_netDICE_Brown import UNetDICE
from model.u_netDICE_BrownCE import UNetDICE_CE
from model.u_net2 import UNet
import timeit
import math
from tensorboardX import SummaryWriter
from PIL import Image
from utils.transforms import vl2im
from utils.misc import Logger
import sys

#from utils.lovasz_losses import lovasz_softmax
import torch.nn.functional as F
from torchmetrics import JaccardIndex, Dice

import kornia as K
from kornia import morphology as morph

from losses import *
import torchgeometry as tgm

start = timeit.default_timer()

IMG_MEAN = np.array((96.25992, 109.307915 , 128.95671), dtype=np.float32)  # BGR

#RESTORE_FROM = '/home/mjp5595/LEAF_UNET_GREEN_SEP22.pth'
#RESTORE_FROM = '/home/mjp5595/LEAF_UNET_GREEN_SEP22_CE.pth'
RESTORE_FROM = ''
BATCH_SIZE = 6
MAX_EPOCH = 200
#MAX_EPOCH = 500
GPU = "0"
root_dir = '/data/leaf_train/green/Sep22/'
DATA_LIST_PATH = root_dir + '/train.txt'
VAL_LIST_PATH = root_dir + '/validation.txt'
INPUT_SIZE = '512,512'
LEARNING_RATE = 2.5 * 10 ** (-4)
MOMENTUM = 0.9
NUM_CLASSES = 2
# fb: freeze BatchNorm
# wl: weighted loss
postfix = "-fb"
#CLASS_DISTRI = [1.0, 1.0]  # [90,272,867, 2,001,821]
CLASS_DISTRI = [12.0, 1.0]  # [92,988,518 2,956,186]
POWER = 0.9
RANDOM_SEED = 1234
#TODO
SAVE_PRED_EVERY = 50
#SNAPSHOT_DIR = root_dir + 'PatchNet/snapshots_Feb23_green_Focal'+postfix
#SNAPSHOT_DIR = root_dir + 'PatchNet/snapshots_Feb23_green_DICE2chan'+postfix
#SNAPSHOT_DIR = root_dir + 'PatchNet/snapshots_Feb23_green_Tversky'+postfix
#SNAPSHOT_DIR = root_dir + 'PatchNet/snapshots_Feb23_green_TverskyFocal'+postfix
#SNAPSHOT_DIR = root_dir + 'PatchNet/snapshots_Feb23_green_IoU'+postfix
#SNAPSHOT_DIR = root_dir + 'PatchNet/snapshots_Feb23_green_CE'+postfix
#SNAPSHOT_DIR = root_dir + 'PatchNet/snapshots_Feb23_green_WeightedCE'+postfix
SNAPSHOT_DIR = root_dir + 'PatchNet/snapshots_Feb23_green_SSIM'+postfix
#SAPSHOT_DIR = root_dir + 'PatchNet/snapshots_Feb23_green_Boundary'+postfix
#SNAPSHOT_DIR = root_dir + 'PatchNet/snapshots_Feb23_green_Lovasz'+postfix
WEIGHT_DECAY = 0.0005
NUM_EXAMPLES_PER_EPOCH = 1016
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
               NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)
#if not os.path.exists(IMGSHOT_DIR):
#    os.makedirs(IMGSHOT_DIR)

LOG_PATH = SNAPSHOT_DIR + "/B"+format(BATCH_SIZE, "04d")+"E"+format(MAX_EPOCH, "04d")+"_NEW.log"
sys.stdout = Logger(LOG_PATH, sys.stdout)
print(DATA_LIST_PATH)
print(VAL_LIST_PATH)
print("num of epoch:", MAX_EPOCH)
print("RESTORE_FROM:", RESTORE_FROM)
print(NUM_EXAMPLES_PER_EPOCH)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="UNet Network")
    parser.add_argument("--set-start", default=False)
    parser.add_argument("--start-step", default=0, type=int)
    parser.add_argument("--is-training", default=False,
                        help="Whether to freeze BN layers, False for Freezing")
    parser.add_argument("--num-workers", default=16)
    parser.add_argument("--final-step", type=int, default=int(NUM_STEPS_PER_EPOCH * MAX_EPOCH),
                        help="Number of training steps.")
    parser.add_argument("--fine-tune", default=False)
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", default=GPU,
                        help="choose gpu device.")
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--save-img-freq', default=200, type=int,
                        metavar='N', help='save image frequency')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the text file listing the images in the dataset.")
    parser.add_argument("--val-list", type=str, default=VAL_LIST_PATH,
                        help="Path to the text file listing the validation images in the dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("-class-distri", default=CLASS_DISTRI)
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", default=True,
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-jitter", default=True)
    parser.add_argument("--random-rotate", default=True)
    parser.add_argument("--random-scale", default=False,
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    return parser.parse_args()


args = get_arguments()

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, actual_step):
    """Original Author: Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(args.learning_rate, actual_step * args.batch_size, MAX_ITER, args.power)
    optimizer.param_groups[0]['lr'] = lr


def resize_target(target, size):
    new_target = np.zeros((target.shape[0], size, size), np.float32)
    for i, t in enumerate(target.numpy()):
        new_target[i, ...] = cv2.resize(t, (size,) * 2, interpolation=cv2.INTER_NEAREST)
    return torch.from_numpy(new_target).long()


def main():
    """Create the model and start the training."""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    print( 'args.gpu :',args.gpu )
    print( 'torch.cuda.device_count() :',torch.cuda.device_count() )
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)

    cudnn.enabled = True
    torch.manual_seed(args.random_seed)

    #model = UNetDICE(args.num_classes)
    #model = UNetDICE_CE(args.num_classes)
    model = UNet(args.num_classes)
    model = torch.nn.DataParallel(model)

    optimizer = optim.SGD(model.parameters(),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    if args.fine_tune:
        saved_state_dict = torch.load(args.restore_from)
        saved_state_dict = saved_state_dict['state_dict']
        new_params = model.state_dict().copy()
        for i in saved_state_dict:
            i_parts = i.split('.')
            if 'final' not in i_parts:
                new_params[i] = saved_state_dict[i]
        model.load_state_dict(new_params)
    elif args.restore_from:
        if os.path.isfile(args.restore_from):
            print("=> loading checkpoint '{}'".format(args.restore_from))
            checkpoint = torch.load(args.restore_from)
            try:
                if args.set_start:
                    args.start_step = int(math.ceil(checkpoint['example'] / args.batch_size))
                model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (step {})"
                      .format(args.restore_from, args.start_step))
            except:
                model.load_state_dict(checkpoint)
                print("=> loaded checkpoint '{}'".format(args.restore_from))
        else:
            print("=> no checkpoint found at '{}'".format(args.restore_from))
            exit(0)

    if not args.is_training:
        # Frozen BN
        # when training, the model will use the running means and the
        # running vars of the pretrained model.
        # But note that eval() doesn't turn off history tracking.
        print("Freezing BN layers")
        model.eval()
    else:
        print("Normal BN layers")
        model.train()
    model.cuda()

    cudnn.benchmark = True

    trainloader = data.DataLoader(LEAFTrain(args.data_list,
                                               scale=args.random_scale,
                                               mirror=args.random_mirror, color_jitter=args.random_jitter,
                                               rotate=args.random_rotate,
                                               ),
                                  batch_size=args.batch_size,
                                  )

    valloader = data.DataLoader(LEAFTrain(args.val_list,
                                               scale=args.random_scale,
                                               mirror=args.random_mirror, color_jitter=args.random_jitter,
                                               rotate=args.random_rotate,
                                               ),
                                  batch_size=args.batch_size,
                                  )

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    writer = SummaryWriter(args.snapshot_dir)

    # weight computation
    #TODO
    class_distri = np.array(args.class_distri)
    normalized_class_distri = class_distri/np.sum(class_distri)
    class_weight = 1 / normalized_class_distri
    print('class_weight :',class_weight)

    cnt = 0
    actual_step = args.start_step
    best_val_loss = np.Inf
    while actual_step < args.final_step:
        iter_end = timeit.default_timer()
        for i_iter, batch in enumerate(trainloader):
            actual_step = int(args.start_step + cnt)

            data_time.update(timeit.default_timer() - iter_end)

            images, labels, patch_name = batch
            labels = labels.type(torch.int64)

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, actual_step)

            pred = model(images)

            #criterion = IoULoss().cuda() # to put on cuda or cpu
            #criterion = TverskyLoss().cuda()
            #criterion = FocalTverskyLoss().cuda()
            #criterion = LovaszLoss().cuda()
            #criterion = BoundaryLoss().cuda()
            criterion = tgm.losses.SSIM(11, reduction='mean').cuda()
            #criterion = tgm.losses.FocalLoss(alpha=0.5, gamma=2.0, reduction='mean').cuda()
            #criterion = FocalLoss(alpha=5,gamma=2.0).cuda()
            #criterion = FocalLoss().cuda() # to put on cuda or cpu
            #criterion = SoftDiceLoss().cuda() # to put on cuda or cpu
            #criterion = torch.nn.CrossEntropyLoss().cuda() # to put on cuda or cpu
            #criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0,12.0])).cuda() # to put on cuda or cpu

            pred = torch.softmax(pred,axis=1)
            labels = torch.nn.functional.one_hot(labels)
            labels = labels.transpose(1,3)
            labels = labels.type(torch.float32)
            loss = criterion(pred,labels.cuda())
            #print('pred.shape :',pred.shape)
            #loss = criterion(pred,labels.cuda())
            #pred = torch.softmax(pred,axis=1)[:,1,:,:]
            #pred = pred.view(b,1,h,w)
            #labels = labels.type(torch.float32)
            #labels = labels.view(b,1,h,w)
            #loss = criterion(pred, labels.cuda())

            losses.update(loss.item(), pred.size(0))

            acc = _pixel_accuracy(pred.data.cpu().numpy(), labels.data.cpu().numpy())
            accuracy.update(acc, pred.size(0))
            
            loss.backward()
            optimizer.step()


            #########################################################################################3
            if actual_step % NUM_STEPS_PER_EPOCH == 0:
                with torch.no_grad():
                    val_loss = 0. 
                    val_acc = 0
                    for val_i_iter, val_batch in enumerate(valloader):

                        val_images, val_labels, patch_name = val_batch
                        val_pred = model(val_images)
                        val_image = val_images.data.cpu().numpy()[0]

                        val_labels = resize_target(val_labels, val_pred.size(2))

                        val_pred = torch.softmax(val_pred,axis=1)
                        val_labels = torch.nn.functional.one_hot(val_labels)
                        val_labels = val_labels.transpose(1,3)
                        val_labels = val_labels.type(torch.float32)
                        v_loss = criterion(val_pred,val_labels.cuda())
                        #v_loss = criterion(val_pred,val_labels.cuda())
                        #v_loss = criterion(torch.softmax(val_pred,axis=1)[:,1,:,:],val_labels.cuda())
                        #val_pred = torch.softmax(val_pred,axis=1)[:,1,:,:]
                        #b,h,w = val_pred.shape
                        #val_pred = val_pred.view(b,1,h,w)
                        #val_labels = val_labels.type(torch.float32)
                        #val_labels = val_labels.view(b,1,h,w)
                        #v_loss = criterion(val_pred, val_labels.cuda())

                        val_loss += v_loss.item()

                        val_acc += _pixel_accuracy(val_pred.cpu().numpy(), val_labels.data.cpu().numpy())
                    
                if val_loss < best_val_loss:
                    print('saving the best model so far ...\tprev best val loss : {}\t curr best val loss : {}'.format(best_val_loss,val_loss))
                    torch.save(model.state_dict(),osp.join(args.snapshot_dir,'best_model.pth'))
                    best_val_loss = val_loss

                writer.add_scalar("train/train_loss", losses.avg, actual_step)
                writer.add_scalar("train/pixel_accuracy", accuracy.avg, actual_step)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], actual_step)
                writer.add_scalar("val/train_loss", val_loss, actual_step)
                writer.add_scalar("val/pixel_accuracy", val_acc, actual_step)
                            
            batch_time.update(timeit.default_timer() - iter_end)
            iter_end = timeit.default_timer()

            if actual_step % args.print_freq == 0:
                print('iter: [{0}]{1}/{2}\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Pixel Accuracy {accuracy.val:.3f} ({accuracy.avg:.3f})'.format(
                    cnt, actual_step, args.final_step, batch_time=batch_time,
                    data_time=data_time, loss=losses, accuracy=accuracy))

            if actual_step >= args.final_step:
                break
            cnt += 1

    end = timeit.default_timer()
    print(end - start, 'seconds')

def _pixel_accuracy(pred, target):
    accuracy_sum = 0.0
    for i in range(0, pred.shape[0]):
        #out = np.where(pred[i]>=0,1,0)
        out = np.argmax(pred[i],axis=0)
        accuracy = np.sum(out == target[i], dtype=np.float32) / out.size
        accuracy_sum += accuracy
    return accuracy_sum / args.batch_size



class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    main()

# mean Dice 0.21209761175529887
# batch normalization
