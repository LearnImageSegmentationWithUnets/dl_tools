#!/usr/bin/env python3

## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

## python infer.py --name cityscapes_test3 --samples-dir data_kitti/testing/image_2 --output-dir cityscapes_test_output --data-source cityscapes

## python infer.py --name seabright_test --samples-dir data_seabright/samples/RGB/val/seabright --output-dir seabright_test_output --data-source seabright

## python infer.py --name ontario_test --samples-dir data_ontario/samples/RGB/val/ontario --output-dir ontario_test_output --data-source ontario

## python infer.py --name elwha_test10 --samples-dir data_elwha/samples/RGB/val/elwha --output-dir elwha_test_output10 --data-source elwha

## python infer_crf.py --name elwha_test100 --samples-dir D:\Elwha\Elwha_20130919\png --output-dir elwha_deploy --data-source elwha

import argparse
import math
import sys
import cv2
import os

import tensorflow as tf
import numpy as np

from fcnvgg import FCNVGG
from utils import *
from glob import glob
from tqdm import tqdm

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels, unary_from_softmax
from scipy.misc import imresize

from skimage.morphology import disk
from skimage.filters.rank import median 


# =========================================================
def getCRF(image, Lc, theta1, theta2, n_iter, label_lines, compat_spat=12, compat_col=40, scale=5, prob=0.5):

#        n_iters: number of iterations of MAP inference.
#        sxy_gaussian: standard deviations for the location component
#            of the colour-independent term.
#        compat_gaussian: label compatibilities for the colour-independent
#            term (can be a number, a 1D array, or a 2D array).
#        kernel_gaussian: kernel precision matrix for the colour-independent
#            term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
#        normalisation_gaussian: normalisation for the colour-independent term
#            (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
#        sxy_bilateral: standard deviations for the location component of the colour-dependent term.
#        compat_bilateral: label compatibilities for the colour-dependent
#            term (can be a number, a 1D array, or a 2D array).
#        srgb_bilateral: standard deviations for the colour component
#            of the colour-dependent term.
#        kernel_bilateral: kernel precision matrix for the colour-dependent term
#            (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
#        normalisation_bilateral: normalisation for the colour-dependent term
#            (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
	  
      H = image.shape[0]
      W = image.shape[1]	
      d = dcrf.DenseCRF2D(H, W, len(label_lines)+1)
      U = unary_from_labels(Lc.astype('int'), len(label_lines)+1, gt_prob= prob)

      d.setUnaryEnergy(U)

      del U

      # This potential penalizes small pieces of segmentation that are
      # spatially isolated -- enforces more spatially consistent segmentations
      # This adds the color-independent term, features are the locations only.
      # sxy = The scaling factors per dimension.
      d.addPairwiseGaussian(sxy=(theta1,theta1), compat=compat_spat, kernel=dcrf.DIAG_KERNEL, #compat=6
                      normalization=dcrf.NORMALIZE_SYMMETRIC)

      # sdims = The scaling factors per dimension.
      # schan = The scaling factors per channel in the image.
      # This creates the color-dependent features and then add them to the CRF
      feats = create_pairwise_bilateral(sdims=(theta2, theta2), schan=(scale, scale, scale), #11,11,11
                                  img=image, chdim=2)

      del image

      d.addPairwiseEnergy(feats, compat=compat_col, #20
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)
      del feats

      Q = d.inference(n_iter)

      #preds = np.array(Q, dtype=np.float32).reshape(
      #  (len(label_lines)+1, nx, ny)).transpose(1, 2, 0)
      #preds = np.expand_dims(preds, 0)
      #preds = np.squeeze(preds)

      return np.argmax(Q, axis=0).reshape((H, W)) #, preds#, p, R, d.klDivergence(Q),

	  
#-------------------------------------------------------------------------------
def sample_generator(samples, image_size, batch_size):
    for offset in range(0, len(samples), batch_size):
        files = samples[offset:offset+batch_size]
        images = []
        names  = []
        for image_file in files:
            image = cv2.resize(cv2.imread(image_file), image_size)
            images.append(image.astype(np.float32))
            names.append(os.path.basename(image_file))
        yield np.array(images), names

#-------------------------------------------------------------------------------
# Parse commandline
#-------------------------------------------------------------------------------
parser = argparse.ArgumentParser(description='Generate data based on a model')
parser.add_argument('--name', default='test',
                    help='project name')
parser.add_argument('--checkpoint', type=int, default=-1,
                    help='checkpoint to restore; -1 is the most recent')
parser.add_argument('--samples-dir', default='test',
                    help='directory containing samples to analyse')
parser.add_argument('--output-dir', default='test-output',
                    help='directory for the resulting images')
parser.add_argument('--batch-size', type=int, default=20,
                    help='batch size')
parser.add_argument('--data-source', default='kitti',
                    help='data source')
args = parser.parse_args()

# args.data = 'seabright'
# args.output_dir = 'seabright_test_output500'
# args.samples_dir = 'data_seabright/samples/RGB/val/seabright'

#-------------------------------------------------------------------------------
# Check if we can get the checkpoint
#-------------------------------------------------------------------------------
state = tf.train.get_checkpoint_state(args.name)
if state is None:
    print('[!] No network state found in ' + args.name)
    sys.exit(1)

try:
    checkpoint_file = state.all_model_checkpoint_paths[args.checkpoint]
except IndexError:
    print('[!] Cannot find checkpoint ' + str(args.checkpoint_file))
    sys.exit(1)

metagraph_file = checkpoint_file + '.meta'

if not os.path.exists(metagraph_file):
    print('[!] Cannot find metagraph ' + metagraph_file)
    sys.exit(1)

#-------------------------------------------------------------------------------
# Load the data source
#-------------------------------------------------------------------------------
try:
    source       = load_data_source(args.data_source)
    label_colors = source.label_colors
except (ImportError, AttributeError, RuntimeError) as e:
    print('[!] Unable to load data source:', str(e))
    sys.exit(1)

#-------------------------------------------------------------------------------
# Create a list of files to analyse and make sure that the output directory
# exists
#-------------------------------------------------------------------------------
samples = glob(args.samples_dir + '/*.png')
if len(samples) == 0:
    print('[!] No input samples found in', args.samples_dir)
    sys.exit(1)

if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)

#-------------------------------------------------------------------------------
# Print parameters
#-------------------------------------------------------------------------------
print('[i] Project name:      ', args.name)
print('[i] Network checkpoint:', checkpoint_file)
print('[i] Metagraph file:    ', metagraph_file)
print('[i] Number of samples: ', len(samples))
print('[i] Output directory:  ', args.output_dir)
print('[i] Image size:        ', source.image_size)
print('[i] # classes:         ', source.num_classes)
print('[i] Batch size:        ', args.batch_size)


n_iter = 40
compat_col = 100
theta1 = 5 ##space
theta2 = 60 ##color
scale = 1
compat_spat = 5 
prob = 0.5
   
#-------------------------------------------------------------------------------
# Create the network
#-------------------------------------------------------------------------------
with tf.Session() as sess:
    print('[i] Creating the model...')
    net = FCNVGG(sess)
    net.build_from_metagraph(metagraph_file, checkpoint_file)

    #---------------------------------------------------------------------------
    # Process the images
    #---------------------------------------------------------------------------
    generator = sample_generator(samples, source.image_size, args.batch_size)
    n_sample_batches = int(math.ceil(len(samples)/args.batch_size))
    description = '[i] Processing samples'

    for x, names in tqdm(generator, total=n_sample_batches,
                        desc=description, unit='batches'):
        feed = {net.image_input:  x,
                net.keep_prob:    1}
        img_labels = sess.run(net.classes, feed_dict=feed)
		
        res = []
        for i in range(len(names)):		
           Lc = 1+img_labels[i,:,:].astype('int')
           Lc[np.random.randint(Lc.shape[0], size=int(Lc.shape[0]/2)), :] = 0
           Lc[:, np.random.randint(Lc.shape[1], size=int(Lc.shape[1]/2))] = 0
           r = getCRF(x[i,:,:,:].astype('uint8'), Lc, theta1, theta2, n_iter, np.arange(source.num_classes), compat_spat, compat_col, scale, prob)		   
           r = median(r, disk(3))		   
           res.append(r)

        res = np.asarray(res).reshape(np.shape(img_labels))		   
		
        imgs = draw_labels_batch(x, res, label_colors, False) #img_labels, label_colors, False)

        for i in range(len(names)):
            cv2.imwrite(args.output_dir + '/' + 'crfmedf_'+ names[i], imgs[i, :, :, :])
print('[i] All done.')
