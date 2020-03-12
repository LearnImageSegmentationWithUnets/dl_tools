#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 14:28:28 2020

@author: bwells
"""
from __future__ import division

import time, socket
import os
import numpy as np
from scipy.io import savemat

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels

from skimage.filters.rank import median
from skimage.morphology import disk

# Change your project parameters here!!
from modules.params_jd import params

import modules.PlotAndSave as PS


def getCRF_justcol(img, Lc, label_lines):
    if np.ndim(img) == 2:
         img = np.dstack((img, img, img))
    H = img.shape[0]
    W = img.shape[1]
    d = dcrf.DenseCRF2D(H, W, len(label_lines) + 1)
    U = unary_from_labels(Lc.astype('int'),
                          len(label_lines) + 1,
                          gt_prob=params['prob'])
    d.setUnaryEnergy(U)
    feats = create_pairwise_bilateral(sdims=(params['theta'], params['theta']),
                                      schan=(params['scale'],
                                             params['scale'],
                                             params['scale']),
                                      img=img,
                                      chdim=2)
    d.addPairwiseEnergy(feats, compat=params['compat_col'],
                        kernel=dcrf.DIAG_KERNEL,
                        normalization=dcrf.NORMALIZE_SYMMETRIC)
    Q = d.inference(params['n_iter'])
    preds = np.array(Q, dtype=np.float32).reshape(
                     (len(label_lines) + 1, H, W)).transpose(1, 2, 0)
    preds = np.expand_dims(preds, 0)
    preds = np.squeeze(preds)

    return np.argmax(Q, axis=0).reshape((H, W)), preds


def DoCrf(o_img, out, params, name, start):
    im = o_img.copy()
    print('Generating dense scene from sparse labels...')
    res, p = getCRF_justcol(im,
                            out.astype('int'),
                            params['classes'])

    resr = median(res, disk(4))

    # TODO: get rid of mat files and use numpy save
    #       Don't do this until seeing where the training
    #       code uses it
    savemat(params['matlab_path'] + name + \
            '_mres.mat',
            {'sparse': out.astype('int'),
              'class': resr.astype('int'),
              'preds': p.astype('float16'),
              'labels': params['classes']},
            do_compression = True)

    Lcorig = out.copy().astype('float')
    Lcorig[Lcorig<1] = np.nan

    PS.PlotAndSave(o_img, resr, out.astype('int'), name, params)

    # ========================================================================
    if os.name == 'posix':  # true if linux/mac
        elapsed = (time.time() - start)
    else:  # windows
        elapsed = (time.clock() - start)
    print("Processing took " + str(elapsed/60) + "minutes")

    # write report
    file = open(params['report_path'] + name + '_report_' +
                socket.gethostname() +
                '.txt', 'w')
    file.write('Image: ' + name + '\n')
    counter = 0
    file.write("Number of Pixels in image =" + \
                str(resr.shape[0] * resr.shape[1]))
    file.write("\n\nClass: percentage of pixels\n")
    for label in params['classes'].keys():
        file.write(label + ': ' +
                    str(np.sum(resr==counter)/\
                        (o_img.shape[0]*o_img.shape[1]))+'\n')
        counter += 1
    file.write('Processing time (mins): ' +
                str(elapsed/60) + '\n')

    file.close()