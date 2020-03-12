#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 13:01:37 2020

@author: bwells

Temporary for testing the Do CRF later function
"""
import os
import time
import numpy as np
from glob import glob
import cv2

import modules.Crf as CR
import modules.WinFuncs as WF

# Change your project parameters here!!
from modules.params_jd import params


def FindIm(fname):
    im_name = glob(params['done_im_pat'] + fname + '.png')  #'*')
    if im_name[0].lower()[-3:] == 'tif':
        img = WF.ReadGeotiff(im_name[0], params['im_order'])
    else:
        img = cv2.imread(im_name[0])
    return img, im_name[0]

def OpenNp(npy_name):
    out = np.load(npy_name)
    return out


if __name__ == "__main__":
    npy_names = glob(params['den_lat_pat'] + '*.npy')
    for i in npy_names:
        name = i[:-4].split(os.sep)[-1]
        im, fip = FindIm(name)
        out = OpenNp(i)
        # start timer
        if os.name == 'posix':  # true if linux/mac or cygwin on windows
            start = time.time()
        else:  # windows
            start = time.clock()
        print(name)
        CR.DoCrf(im, out, params, name, start)
        os.remove(i)
        print(i, "deleted")
