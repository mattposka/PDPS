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
import timeit
import math
from tensorboardX import SummaryWriter
from PIL import Image
from utils.transforms import vl2im
from utils.misc import Logger
import sys

import kornia as K
from kornia import morphology as morph

start = timeit.default_timer()

IMG_MEAN = np.array((96.25992, 109.307915 , 128.95671), dtype=np.float32)  # BGR

RESTORE_FROM = None
BATCH_SIZE = 6
MAX_EPOCH = 200
#MAX_EPOCH = 500
GPU = "2"
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
SNAPSHOT_DIR = root_dir + 'PatchNet/snapshots_Sep22_Brown'+postfix
IMGSHOT_DIR = root_dir + 'PatchNet/imgshots_Sep22_Brown'+postfix
WEIGHT_DECAY = 0.0005
NUM_EXAMPLES_PER_EPOCH = 1016
NUM_STEPS_PER_EPOCH = math.ceil(NUM_EXAMPLES_PER_EPOCH / float(BATCH_SIZE))
MAX_ITER = max(NUM_EXAMPLES_PER_EPOCH * MAX_EPOCH + 1,
               NUM_STEPS_PER_EPOCH * BATCH_SIZE * MAX_EPOCH + 1)
if not os.path.exists(SNAPSHOT_DIR):
    os.makedirs(SNAPSHOT_DIR)
if not os.path.exists(IMGSHOT_DIR):
    os.makedirs(IMGSHOT_DIR)

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
    parser.add_argument("--img-dir", type=str, default=IMGSHOT_DIR,
                        help="Where to save images of the model.")
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

class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets): # pred,label
        smooth = 1
        num = targets.size(0)
        """
       I am assuming the model does not have sigmoid layer in the end. if that is the case, change torch.sigmoid(logits) to simply logits
        """
        targets = Variable(targets.float()).cuda()
        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


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

    model = UNetDICE(args.num_classes)
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
    val_loss_list = []
    train_loss_list = []
    val_acc_list = []
    train_acc_list = []
    while actual_step < args.final_step:
        iter_end = timeit.default_timer()
        for i_iter, batch in enumerate(trainloader):
            actual_step = int(args.start_step + cnt)

            data_time.update(timeit.default_timer() - iter_end)

            images, labels, patch_name = batch

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, actual_step)

            pred = model(images)
            labels = resize_target(labels, pred.size(2))

            criterionSDL = SoftDiceLoss().cuda() # to put on cuda or cpu
            lossSDL = criterionSDL(pred, labels)

            losses.update(lossSDL.item(), pred.size(0))

            acc = _pixel_accuracy(pred.data.cpu().numpy(), labels.data.cpu().numpy())
            accuracy.update(acc, pred.size(0))
            
            lossSDL.backward()
            optimizer.step()

            #########################################################################################3
            STOP_EARLY = False
            EPOCHS_BEFORE_STOPPING = 8
            if actual_step % NUM_STEPS_PER_EPOCH == 0:
                with torch.no_grad():
                    val_loss = 0. 
                    for val_i_iter, val_batch in enumerate(valloader):

                        val_images, val_labels, patch_name = val_batch
                        val_pred = model(val_images)
                        val_image = val_images.data.cpu().numpy()[0]

                        val_labels = resize_target(val_labels, val_pred.size(2))

                        v_lossSDL = criterionSDL(val_pred, val_labels)

                        v_loss = v_lossSDL
                        val_loss += v_loss.item()

                        val_acc = _pixel_accuracy(val_pred.cpu().numpy(), val_labels.data.cpu().numpy())
                        
                val_loss_list.append(val_loss)
                train_loss_list.append(losses.val)
                val_acc_list.append(val_acc)
                train_acc_list.append(accuracy.val)
                
                #if len(val_loss_list) > EPOCHS_BEFORE_STOPPING:
                #    if val_loss_list[-1] > val_loss_list[-1*(EPOCHS_BEFORE_STOPPING+1)]:
                #        STOP_EARLY = True
                #        print('STOPPING EARLY!!!!!!!')
                #    print('Val Loss :',val_loss_list[-1*(EPOCHS_BEFORE_STOPPING+1):])
            #########################################################################################3

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
                writer.add_scalar("train_loss", losses.avg, actual_step)
                writer.add_scalar("pixel_accuracy", accuracy.avg, actual_step)
                writer.add_scalar("lr", optimizer.param_groups[0]['lr'], actual_step)

                val_loss_list_array = np.array(val_loss_list)
                train_loss_list_array = np.array(train_loss_list)
                val_acc_list_array = np.array(val_acc_list)
                train_acc_list_array = np.array(train_acc_list)

                np.savetxt(args.snapshot_dir+'val_loss_log.txt',val_loss_list_array)
                np.savetxt(args.snapshot_dir+'train_loss_log.txt',train_loss_list_array)
                np.savetxt(args.snapshot_dir+'val_acc_log.txt',val_acc_list_array)
                np.savetxt(args.snapshot_dir+'train_acc_log.txt',train_acc_list_array)


            #if actual_step % args.save_img_freq == 0:
            #    msk_size = pred.size(2)
            #    image = np.transpose(image,(1, 2, 0))
            #    image = cv2.resize(image, (msk_size, msk_size), interpolation=cv2.INTER_NEAREST)
            #    image2show = image[:,:,:3]
            #    image2show = image2show*255
            #    label = labels.data.cpu().numpy()[0]
            #    label = vl2im(label)


            #    #single_pred = pred.data.cpu().numpy()[0].argmax(axis=0)
            #    #single_pred = vl2im(single_pred)
            #    single_pred = np.where(pred.data.cpu().numpy()[0]>=0,1,0)
            #    #print('single_pred.shape :',single_pred.shape)
            #    single_pred = vl2im(single_pred)

            #    new_im = Image.new('RGB', (msk_size * 3, msk_size))
            #    new_im.paste(Image.fromarray(image2show.astype('uint8'), 'RGB'), (0, 0))
            #    new_im.paste(Image.fromarray(single_pred.astype('uint8'), 'RGB'), (msk_size, 0))
            #    new_im.paste(Image.fromarray(label.astype('uint8'), 'RGB'), (msk_size * 2, 0))
            #    new_im_name = 'B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '_' + patch_name[0].replace('.npy','.png')
            #    new_im_file = os.path.join(args.img_dir, new_im_name)
            #    new_im.save(new_im_file)

            if actual_step % args.save_pred_every == 0 and cnt != 0:
                print('taking snapshot ...')
                torch.save({'example': actual_step * args.batch_size,
                            'state_dict': model.state_dict()},
                           osp.join(args.snapshot_dir,
                                    'LEAF_UNET_B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '.pth'))
            if actual_step >= args.final_step:
                break
            if STOP_EARLY == True:
                break
            cnt += 1
        if STOP_EARLY == True:
            break


    print('save the final model ...')
    torch.save({'example': actual_step * args.batch_size,
                'state_dict': model.state_dict()},
               osp.join(args.snapshot_dir,
                        'LEAF_UNET_B' + format(args.batch_size, "04d") + '_S' + format(actual_step, "06d") + '_FINAL.pth'))

    end = timeit.default_timer()
    print(end - start, 'seconds')

def _pixel_accuracy(pred, target):
    accuracy_sum = 0.0
    for i in range(0, pred.shape[0]):
        out = np.where(pred[i]>=0,1,0)
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