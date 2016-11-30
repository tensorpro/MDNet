import numpy as np
from load import crop, roi_sample, overlap_ratio, intbb
import matplotlib.pyplot as plt



def show_bb(img, bb, color):
    """
    Shows image with `bb` drawn on it.
    """
    img = img.copy()
    plt.imshow(draw_bbox(bb,img,color))
    plt.show()

def demo_gen(roidb,n=4):
    """
    Shows how the generator samples from roidb.
    """
    gen = generator(roidb,4,2, show=True)
    for _ in range(n):
        (D,img), label = gen.next()
        plt.imshow(img)
        plt.show()


def crop_demo(img, bb, numpos=5, numneg=10):
    """
    Shows some example crops of positive and negative examples
    """
    pos = roi_sample(bb, img.shape, False, trans_range=.1,
                     scale_range=5, overlap_range=[.7, 1])
    img = img.copy()
    for _ in range(numpos):
        smp = roi_sample(bb, img.shape, False, trans_range=.1,
                         scale_range=5, overlap_range=[.7, 1])
        plt.imshow(crop(img, smp))
        plt.show()

    for _ in range(numneg):
        smp = roi_sample(bb, img.shape, False, trans_range=2,
                         scale_range=10, overlap_range=[0, .5])
        plt.imshow(crop(img, smp))
        plt.show()

def draw_bbox(bbox, img, color=[255, 0, 255]):
    """
    Draws bbox on image. 
    """
    left, top, width, height = intbb(bbox)
    right = min(left + width, img.shape[1] - 1)
    bottom = min(top + height, img.shape[0] - 1)
    img[top, left:right] = color
    img[bottom, left:right] = color
    img[top:bottom, left] = color
    img[top:bottom, right] = color
    return img


def track(bboxs, imgs):
    """
    Shows bb across sequence.
    """
    for bbox, img in zip(bboxs, imgs):
        plt.imshow(draw_bbox(bbox, img))
        plt.show()


def show_samps(img, bb, valid=True, n=5):
    """
    Samples `n` times from img around `bb`
    and colors:

    positive examples: green
    negative examples: red
    discarded examples: black
    """
    bb = np.round(bb).astype(int)
    plt.imshow(draw_bbox(bb, img))
    img = draw_bbox(bb, img)
    plt.show()
    for _ in range(n):
        smp = roi_sample(bb, img.shape, valid,
                         overlap_range=(.7, 1)).astype(int)
        r = (overlap_ratio(smp, bb))
        if .7 <= r <= 1:
            color = green
        elif .1 <= r <= .5:
            color = red
        else:
            color = [0] * 3
        plt.imshow(draw_bbox(smp, img, color))
        plt.show()


def sample_demo(img, bb, numpos=5, numneg=10):
    """
    Outputs an image showing the positive and negative
    examples
    """
    img = img.copy()
    output_img = draw_bbox(bb, img)
    for _ in range(numpos):
        smp = roi_sample(bb, img.shape, False, trans_range=.1,
                         scale_range=5, overlap_range=[.7, 1])
        ouput_img = draw_bbox(smp, output_img, color=green)
    for _ in range(numneg):
        smp = roi_sample(bb, img.shape, False, trans_range=2,
                         scale_range=10, overlap_range=[0, .5])
        ouput_img = draw_bbox(smp, output_img, color=red)
    plt.imsave("samples", output_img)
