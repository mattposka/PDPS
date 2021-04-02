# Testing U-Net for LEAF Dataset
# Track set True for Normal BN (Frozen BN when training) or False for batch stats
import argparse
import numpy as np
import time
import torch
from torch.autograd import Variable
from torch.utils import data
from utils.datasets import LEAFTest
from model.u_net import UNet
import os
import torch.nn as nn
from scipy.sparse import save_npz, coo_matrix

IMG_MEAN = np.array((62.17962105572224, 100.62603236734867, 131.60830906033516), dtype=np.float32)  # BGR
version = "200717"
# fb: freeze BatchNorm
# wl: weighted loss
postfix = "-fb"
test_dir = os.path.join("/data/AutoPheno/green/", version)
# "/data/AutoPheno/green/"
train_dir = os.path.join("/data/AutoPheno/green/", version)
# "/data/AutoPheno/green/"
DATA_DIRECTORY = os.path.join(test_dir, '512_test_stride_64', 'images')
# test_dir + '512_test_stride_64/images'
DATA_LIST_PATH = os.path.join(test_dir, '512_s64')
# test_dir + '512_s64/'
if not os.path.exists(DATA_LIST_PATH):
    os.mkdir(DATA_LIST_PATH)
NUM_CLASSES = 2
model_name = "21700"
RESTORE_FROM = '/data/AutoPheno/green/200723/PatchNet/snapshots-fb/LEAF_UNET_B0064_S021700.pth'
# train_dir + '/PatchNet/snapshots' + postfix + '/LEAF_UNET_B0064_S' + format(int(model_name), "06d") + '.pth'
NPZ_PATH = test_dir + '/PatchNet/npz' + postfix + '/' + model_name
if not os.path.exists(NPZ_PATH):
    os.makedirs(NPZ_PATH)

MAP_PATH = test_dir + '/unet/map' + postfix + '/' + model_name
if not os.path.exists(MAP_PATH):
    os.makedirs(MAP_PATH)
BATCH_SIZE = 32
INPUT_SIZE = (512, 512)
LOG_PATH = test_dir + '/unet/logfiles' + postfix
if not os.path.exists(LOG_PATH):
    os.makedirs(LOG_PATH)


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="UNet Network")
    parser.add_argument("--npz-path", default=NPZ_PATH)
    parser.add_argument("--map-path", default=MAP_PATH)
    parser.add_argument("--track-running-stats", default=True) # set false to use current batch_stats when eval
    parser.add_argument("--momentum", default=0) # set 0 to freeze running mean and var, useless when eval
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", default='0',
                        help="choose gpu device.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument('--print-freq', '-p', default=5, type=int,
                        metavar='N', help='print frequency')
    parser.add_argument('--log-path', default=LOG_PATH)
    return parser.parse_args()


def main():
    preprocess_start_time = time.time()
    """Create the model and start the evaluation process."""
    args = get_arguments()
    print("Restored from:", args.restore_from)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    LogName = "Test_HeatMap_log.txt"
    LogFile = os.path.join(args.log_path, LogName)
    log = open(LogFile, 'w')
    log.writelines('batch size:' + str(args.batch_size) + ' ' + 'gpu:' + args.gpu + '\n')
    log.writelines(args.data_list + '\n')
    log.writelines('restore from ' + args.restore_from + '\n')

    model = UNet(args.num_classes)
    model = nn.DataParallel(model)
    model.cuda()
    saved_state_dict = torch.load(args.restore_from)
    num_examples = saved_state_dict['example']
    if args.track_running_stats:
        print("using running mean and running var")
        log.writelines("using running mean and running var\n")
        model.load_state_dict(saved_state_dict['state_dict'])
    else:
        print("using current batch stats instead of running mean and running var")
        log.writelines("using current batch stats instead of running mean and running var\n")
        print("if you froze BN when training, maybe you are wrong now!!!")
        log.writelines("if you froze BN when training, maybe you are wrong now!!!\n")
        new_params = saved_state_dict['state_dict'].copy()
        for i in saved_state_dict['state_dict']:
            i_parts = i.split('.')
            if ("running_mean" in i_parts) or ("running_var" in i_parts):
                del new_params[i]
        model.load_state_dict(new_params)

    model.eval()
    # model.train()
    log.writelines('preprocessing time: ' + str(time.time() - preprocess_start_time) + '\n')

    TestDir = os.listdir(args.data_dir)
    TestDir.sort()
    for TestName in TestDir:
        print('Processing '+TestName)
        log.writelines('Processing ' + TestName + '\n')

        TestTxt = os.path.join(args.data_list, TestName + '.txt')
        testloader = data.DataLoader(LEAFTest(TestTxt, crop_size=INPUT_SIZE, mean=IMG_MEAN),
                                     batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        TestNpzPath = os.path.join(args.npz_path, TestName)
        TestMapPath = os.path.join(args.map_path, TestName)
        if not os.path.exists(TestNpzPath):
            os.mkdir(TestNpzPath)
        if not os.path.exists(TestMapPath):
            os.mkdir(TestMapPath)

        batch_time = AverageMeter()
        with torch.no_grad():
            end = time.time()
            for index, (image, name) in enumerate(testloader):

                output = model(Variable(image).cuda())
                del image
                Softmax = torch.nn.Softmax2d()
                pred = torch.max(Softmax(output), dim=1, keepdim=True)
                del output

                for ind in range(0, pred[0].size(0)):
                    prob = torch.squeeze(pred[0][ind]).data.cpu().numpy()
                    prob = coo_matrix(prob)
                    if len(prob.data) == 0:
                        continue
                    mapname = name[ind].replace('.jpg', '_N' + str(num_examples) + '_MAP.npz')
                    mapfile = os.path.join(TestMapPath, mapname)
                    save_npz(mapfile, prob.tocsr())

                    msk = torch.squeeze(pred[1][ind]).data.cpu().numpy()
                    msk = coo_matrix(msk)
                    if len(msk.data) == 0:
                        continue
                    npzname = name[ind].replace('.jpg', '_N' + str(num_examples) + '_MSK.npz')
                    npzfile = os.path.join(TestNpzPath, npzname)
                    save_npz(npzfile, msk.tocsr())

                batch_time.update(time.time() - end)
                end = time.time()

                if index % args.print_freq == 0:
                    print('Test:[{0}/{1}]\t'
                          'Time {batch_time.val:.3f}({batch_time.avg:.3f})'
                          .format(index, len(testloader), batch_time=batch_time))

        print('The total test time for '+TestName+' is '+str(batch_time.sum))
        log.writelines('batch num:' + str(len(testloader)) + '\n')
        log.writelines('The total test time for ' + TestName + ' is ' + str(batch_time.sum) + '\n')
    log.writelines('The total running time is '+str(time.time()-preprocess_start_time)+'\n')
    log.close()


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
