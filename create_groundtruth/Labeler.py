# -*- coding: utf-8 -*-
# from __future__ import division
import time
import ctypes
import subprocess
import os
import cv2
import numpy as np
from glob import glob
import modules.CWinFuncs as WF
import modules.Crf as CR
from modules.CWinFuncs import MaskPainter

# Change your project parameters here!!
from modules.params_jd import params


def FindNextIm(params):
    # Find the next image in images_path with any of the defined image types
    imp = [glob(e) for e in params['image_types']]
    images = []
    for i in imp:
        images = [*images, *i]

    image_path = images[0]
    name, ext = os.path.splitext(image_path)
    name = name.split(os.sep)[-1]
    return name, image_path


def TimeScreen():
    """
    Starts a timer and gets the screen size. I realize these should be seperate
      functions, but the os name is here, so might as well. Maybe this should
      seperated in the future.
    Takes:
        nothing
    Returns:
        start : datetime stamp
        screen_size : tuple of screen size
    """

    # start timer and get screen size
    if os.name == 'posix':  # true if linux/mac or cygwin on windows
        start = time.time()
        cmd = ['xrandr']
        cmd2 = ['grep', '*']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
        p2 = subprocess.Popen(cmd2, stdin=p.stdout, stdout=subprocess.PIPE)
        p.stdout.close()
        resolution_string, junk = p2.communicate()
        resolution = resolution_string.split()[0].decode("utf-8")
        width, height = resolution.split('x')
        screen_size = tuple((int(height), int(width)))
    else:  # windows
        start = time.clock()
        user32 = ctypes.windll.user32
        screen_size = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    return start, screen_size


def OpenImage(image_path, params):
    """
    Opens the image of any type I know of
    Returns the image in numpy array format
    Takes:
        image_path : string
            Full or relative path to image
        params : dict
            Dictionary of parameters set in the parameters file
    Returns:
        numpy array of image 2D or 3D #NOTE: want this to do multispectral
    """
    if image_path.lower()[-3:] == 'tif':
        img = WF.ReadGeotiff(image_path, params['im_order'])
    else:
        img = cv2.imread(image_path)
    return img


if __name__ == "__main__":
    name, image_path = FindNextIm(params)
    start, screen_size = TimeScreen()
    o_img = OpenImage(image_path, params)
    # Get image mean
    # NOTE: this is probably not useful, needs to be moved at the least
    im_mean = np.mean(o_img.copy()).astype('int')

    # Check if the image is too large for this program and split it
    if WF.SplitCheck(o_img.shape):
        WF.SplitImage(o_img,
                      name,
                      image_path.lower()[-4:],
                      params)
        print("Image was split, please rerun the program for next image")
        raise SystemExit
    else:
        print(name)
        mp = MaskPainter(o_img.copy(), params, screen_size, im_mean)
        out = mp.LabelWindow()
        if not params['do_dense_later']:
            CR.DoCrf(o_img.copy(), out, params, name, start)
            os.rename(image_path,
                      image_path.replace(params['images_path'][:-1],
                                         params['done_im_pat'][:-1]))
        else:
            np.save(params['den_lat_pat'] + name + '.npy', out)
            os.rename(image_path,
                      image_path.replace(params['images_path'][:-1],
                                         params['done_im_pat'][:-1]))
            print("NPY file saved in", params['den_lat_pat'],
                  "\nImage moved to", params['done_im_pat'])
            print("All set to run CRFLater.py or run Labeler again to" +
                  " label the next one")
