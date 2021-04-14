# Merging the segmentation results of each patch
# and outputing the entire segmentation map for LEAF
# Author: Haomiao Ni
# Note that You can merge and segment patches at the same time
import os
from time import sleep
import numpy as np
import threading
from scipy.sparse import *
from scipy.misc import imread

mutex = threading.Lock()
coomats = []
jpgdict = {}


def merge_npz_thread(dirpath, npzfiles, thread_index, ranges, width, height):
    for s in range(ranges[thread_index][0], ranges[thread_index][1]):
        npz = npzfiles[s]
        if npz.split('.')[-1] != 'npz':
            continue
        lefttopx = int(npz.split('_')[-4])
        lefttopy = int(npz.split('_')[-3])
        npzfile = load_npz(os.path.join(dirpath, npz))
        npzfile = npzfile.todense()
        npzfile = coo_matrix(npzfile)
        for i in range(len(npzfile.data)):
            npzfile.row[i] = lefttopy + npzfile.row[i]
            npzfile.col[i] = lefttopx + npzfile.col[i]
            if npzfile.row[i] >= height or npzfile.col[i] >= width:
                npzfile.row[i] = 0
                npzfile.col[i] = 0
                npzfile.data[i] = 0
        npzfile = coo_matrix((npzfile.data, (npzfile.row, npzfile.col)), shape=(height, width))
        global coomats, mutex
        mutex.acquire()
        coomats.append(npzfile)

        mutex.release()
        #print('finish ' + npz + ' \t' + str(s) + '/' + str(len(npzfiles)))
        del npzfile


def merge_npz(npzpath, dir, npzname, width, height):
    dirpath = os.path.join(npzpath, dir)
    npzfileslist = [os.listdir(dirpath)]
    num_npz = len(npzfileslist[0])
    segment = 1
    if num_npz > 1000:
        segment = 8
        l = (num_npz / segment)
        npzfileslist = [npzfileslist[0][l * i:min(num_npz, l * (i + 1))] for i in range(segment)]

    dokmat = dok_matrix((height, width))  # sum of pixels
    dokdict = dok_matrix((height, width))  # number

    for seg in range(segment):
        npzfiles = npzfileslist[seg]
        num_threads = len(npzfiles)
        spacing = np.linspace(0, len(npzfiles), num_threads + 1).astype(np.int)
        ranges = []
        for i in range(len(spacing) - 1):
            ranges.append([spacing[i], spacing[i + 1]])
        threads = []
        global coomats
        coomats = []
        for thread_index in range(len(ranges)):
            args = (dirpath, npzfiles, thread_index, ranges, width, height)
            t = threading.Thread(target=merge_npz_thread, args=args)
            t.setDaemon(True)
            threads.append(t)

        for t in threads:
            t.start()

        # Wait for all the threads to terminate.
        for t in threads:
            t.join()

        for thread_index in range(len(ranges)):
            #print('thread_index = ', thread_index, len(ranges))
            for i in range(len(coomats[thread_index].data)):
                if coomats[thread_index].data[i] > 1:
                    print( '>1 !!' )
                    print( 'coomats[thread_index].data[i] :',coomats[thread_index].data[i] )
                dokmat[coomats[thread_index].row[i], coomats[thread_index].col[i]] += coomats[thread_index].data[i]
                dokdict[coomats[thread_index].row[i], coomats[thread_index].col[i]] += 1
        coomats = []

    # What is the difference between coomat and coodict?
    coomat = coo_matrix(dokmat)
    coodict = dokdict

    del dokmat, dokdict
    for i in range(len(coomat.data)):
        r = coomat.row[i]
        c = coomat.col[i]
        # This section takes care of overlapping patches 
        if coodict[r, c] > 1:
            pos_num = coodict[r, c]
            neg_num = coomat.data[i] - pos_num
            coomat.data[i] = 1 if pos_num > neg_num else 0

    if np.max( coomat ) > 1:
        print( 'np.max( coomat ) :',np.max( coomat ) )
    if np.sum( coomat ) < 20:
        print( 'np.sum( coomat ) :',np.sum( coomat ) )
#    print('saving ' + npzname)
    save_npz(npzname, coomat.tocsr())


def merge_mul_npz_fun(srcpath, path, model_id, logfile, slidedir="imgs_all"):
    npzpath = os.path.join(path, model_id)
    savenpz = os.path.join(path, 'whole_npz/' + model_id)
    if not os.path.exists(savenpz):
        os.makedirs(savenpz)
    listdir = os.listdir(npzpath)
    listdir.sort()
    if len(listdir) != dirlen:
        listdir = listdir[:-1]
    if len(listdir) == 0:
        return
    cnt = 0
    for dir in listdir:
        npzname = os.path.join(savenpz, dir + '_Map.npz')
        if os.path.exists(npzname):
            cnt += 1
            continue
        print(npzname)
        logfile.writelines(npzname)
        img = imread(os.path.join(srcpath+slidedir, dir + '.png'))
        width, height = img.shape[1], img.shape[0]

        merge_npz(npzpath, dir, npzname, int(width), int(height))
    if cnt == dirlen:
        return True


if __name__ == '__main__':
    dirlen = 100  # the total number of testing dataset
    tumorname = "green"
    patchsource = '512_test_stride_64'
    srcpath = '/data/AutoPheno/'
    version = "200717"
    postfix = "-fb"
    slidedir = "random_test_imgs"
    jpgpath = os.path.join(srcpath, tumorname, version, patchsource, 'images')
    jpgdir = os.listdir(jpgpath)
    jpgdir.sort()
    for dir in jpgdir:
        jpgnum = len(os.listdir(os.path.join(jpgpath, dir)))
        jpgdict[dir] = jpgnum

    model_id = '21700'  # 'BACH_UNET_B2_S110000_Frozen_BN_test2048'
    path = os.path.join(srcpath, tumorname, version, 'PatchNet', 'npz'+postfix)
    #srcpath + tumorname+'/'+'PatchNet/npz'
    logdir = os.path.join(srcpath, tumorname, version, 'PatchNet', 'logfiles')
    #srcpath + tumorname+'/'+'PatchNet/logfiles/'
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    logfile = open(os.path.join(logdir, 'merge_npz_' + model_id + postfix + '.log'), 'w')
    while True:
        final = merge_mul_npz_fun(srcpath, path, model_id, logfile, slidedir)
        print('sleep 300s')
        if not final:
            sleep(300)
        else:
            break
