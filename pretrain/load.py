from __future__ import print_function, division
import wget
import wgetter
from zipfile import ZipFile
from os.path import isfile, join
from os import listdir
from time import sleep
from itertools import repeat, count, islice, cycle
from random import random, choice
import numpy as np
import pandas as pd
from skimage.io import imread, imshow, imshow_collection
from skimage.transform import resize


data_path = '../data/'
green = [0, 255, 0]
red = [255, 0, 0]
blue = [0, 0, 255]


def vot_url(year):
    """Gets the url to load the VOT dataset for a given year"""
    return r"http://box.vicos.si/vot/vot{0}.zip".format(year)


def download_vot(years=[2013, 2014, 2015], out_dir="../Data"):
    """
    Dowloads VOT for given years and places then into the
    datasets directory.
    """

    for year in years:
        url = vot_url(year)
        print("Downloading VOT {0}".format(year))
        wgetter.download(url, outdir=out_dir)


def unzip_files(zip_dir):
    """
    Unzips the files from a given directory.
    """
    zip_names = [f for f in listdir(zip_dir) if '.zip' in f]
    for zn in zip_names:
        print('Extracting files for {}'.format(zn))
        zf = ZipFile(join(zip_dir, zn))
        name, extension = zn.split(".")
        zf.extractall(join(zip_dir, name))


def load_seq(dataset, name):
    """
    Loads sequence called `name` from `dataset`.
    """
    fullpath = join(data_path, dataset, name)
    bbox = pd.read_csv(join(fullpath, 'groundtruth.txt'))
    img_names = [join(fullpath, f) for f in listdir(fullpath) if '.jpg' in f]
    imgs = np.array([imread(f) for f in img_names])
    return standardize_bbox(bbox.values), imgs


def intbb(bb):
    """
    Ensures bb has integer values.
    """
    return np.round(bb).astype(int)


def standardize_bbox(bbox):
    """
    Standardize bbox representation across different years of VOT
    VOT2013 had 4 values, and vot 2014/2015 had 8.
    """
    bbox = np.round(bbox).astype(int)
    if len(bbox[0]) == 4:
        return bbox
    elif len(bbox[0]) == 8:
        left, right, top, bottom = bbox[:, [0, 6, 3, 1]].T
        width = right - left
        height = bottom - top
        return np.array([left, top, width, height]).T
    else:
        raise ValueError("bbox must have `len` 4 or 8")


def roi_sample(bb, image_size, valid, scale_factor=1.05, trans_range=.1,
               scale_range=3, overlap_range=[-np.inf, np.inf], max_tries=4):
    """
    Single sample for a given bounding box and image_size
    """
    h, w, _ = image_size
    left, top, width, height = np.round(bb)

    found = False
    min_r, max_r = overlap_range
    tries = 0
    while not found:
        sample = np.array([left + width / 2, top + height / 2, width, height])
        sample[:2] += trans_range * bb[2:] * (np.random.random(2) * 2 - 1)
        to_mul = np.power(scale_factor, scale_range * np.random.random())
        sample[2:] = np.multiply(
            sample[2:], [scale_factor**(scale_range * random() * 2 - 1)] * 2)
        sample[2] = max(5, min(w - 5, sample[2]))
        sample[3] = max(5, min(h - 5, sample[3]))
        sample[:2] -= sample[2:] / 2

        if valid:
            sample[0] = max(1, min(w - sample[2], sample[0]))
            sample[1] = max(1, min(h - sample[3], sample[1]))
        else:
            sample[0] = max(1 - sample[2] / 2,
                            min(w - sample[2] / 2, sample[0]))
            sample[1] = max(1 - sample[3] / 2,
                            min(h - sample[3] / 2, sample[1]))
        r = overlap_ratio(bb, sample)
        tries+=1
        allpos = np.all(sample>=0)
        found = min_r <= r <= max_r and allpos
        if tries >= max_tries and allpos:
            found = True
    return np.round(sample).astype(int)




def crop(img, bb, outshape=[107, 107]):
    """
    Crops the `bb` from `img`, and resizes to `outshape`
    """
    left, top, width, height = bb
    crop=img[top:top + height, left:left + width]
    return resize(crop, outshape)


def overlap(rect1, rect2):
    """
    Returns overlapping area between r1, r2
    Both rectangles are represent x1,y1,width,height
    """
    l1, t1, w1, h1 = rect1
    l2, t2, w2, h2 = rect2
    r1, r2 = l1 + w1, l2 + w2
    b1, b2 = t1 + h1, t2 + h2
    dx = min(r1, r2) - max(l1, l2)
    dy = min(b1, b2) - max(t1, t2)
    return max(dx * dy, 0)


def overlap_ratio(r1, r2):
    """
    Returns overlapping ratio between r1, r2
    Both rectangles are represent x1,y1,width,height
    """
    intersect = overlap(r1, r2)
    w1, h1 = r1[2:]
    w2, h2 = r2[2:]
    return intersect / (w1 * h1 + w2 * h2 - intersect)


def load_seqs(dataset, seqsfile, seqsdir='../SeqLists'):
    """
    loads the sequences in seqsfile from a given dataset
    """
    f = open(join(seqsdir, seqsfile), 'r')
    seqnames = f.read().strip().split('\n')
    seqs = [zip(*load_seq(dataset, sn)) for sn in seqnames]
    return seqs


def create_roidb(toload):
    """
    Creates RoiDB, where there is a list of (bounding box, img) tuples
    for each sequence
    """
    roidb = []
    for (dataset, seqsfile) in toload:
        roidb.extend(load_seqs(dataset, seqsfile))
    return roidb


def generator(roidb, batchsize=128, minisize=8, posprob=1 / 3, show=False):
    """
    Helper generator for batch_generator.
    
    After yielding `batchsize` elements, it moves on to the next
    sequence in roidb

    Samples examples that are positive with probability `posprob`
    
    If `show` is True, will display bounding box
    """
    for (D, seq) in cycle(enumerate(roidb)):
        for _ in range(int(batchsize / minisize)):
            bb, img = choice(seq)
            for _ in range(minisize):
                if random() < posprob:
                    label = 1
                    smp= roi_sample(bb, img.shape, False, trans_range=.1,
                                        scale_range=5, overlap_range=[.7, 1])
                else:
                    label=0
                    smp =roi_sample(bb, img.shape, False, trans_range=2,
                                        scale_range=10, overlap_range=[1e-7, .5])
                if show:
                    show_bb(img, smp, color = green if label is 1 else red)
                val = [0,0]
                val[label]=1
                yield D, crop(img,smp), label

                
def batch_generator(roidb, batchsize=128, minisize=8, posprob=1 / 3, show=False):
    "generates a batch of cropped images"
    gen = generator(roidb, batchsize, minisize, 1 / 3, show)
    for _ in count():
        batch_vals = zip(*islice(gen, batchsize))
        ad, imgs, labs = batch_vals
        yield np.array(imgs), np.array(labs)
        # yield [np.array(imgs), np.array(list(ad))], np.array(labs)
        

to_load = [('vot2013','vot13-otb.txt'),('vot2014','vot14-otb.txt')]
