#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 13:08:42 2020

@author: bwells
"""
import numpy as np
import os
import cv2
import matplotlib
import random
import rasterio
from modules.PlotAndSave import FlipAxForMPL as fip


def WinScales(screen_size, im_size):
    """
    Makes new image size

    Parameters
    ----------
    screen_size : tuple
        Size of screen. Currently being being found in Labeler, but should
        probably be here
    im_size : tuple
        Size of image

    Returns
    -------
    new_im_size : tuple
        Gets the largest size of image (x or y), and finds the scale factor to
        make it the size of the screen. multiplies both directions by that.

    """
    max_dm = np.max(im_size)
    max_s_in_that_direction = screen_size[np.where(im_size == max_dm)[0][0]]
    scale_factor = .9 * max_s_in_that_direction / max_dm
    new_im_size = tuple((int(im_size[1] * scale_factor),
                         int(im_size[0] * scale_factor)))
    return new_im_size


def ReadGeotiff(image_path, rgb):
    """
    This function reads image in GeoTIFF format.
    TODO: Fill in the doc string better

    Parameters
    ----------
    image_path : TYPE
        DESCRIPTION.
    rgb : TYPE
        DESCRIPTION.

    Returns
    -------
    img : TYPE
        DESCRIPTION.

    """
    with rasterio.open(image_path) as src:
        layer = src.read()

    if layer.shape[0] == 3:
        r, g, b = layer
        if rgb == 'RGB':
            img = np.dstack([r, g, b])
        else:
            img = np.dstack([b, g, r])
    elif layer.shape[0] == 4:
        r, g, b, gd = layer
        if rgb == 'RGB':
            img = np.dstack([r, g, b])
        else:
            img = np.dstack([b, g, r])
    # TODO: I have not tested any of the rest of this project for one layer
    else:
        img = layer

    if np.max(img) > 255:
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype('uint8')

    img[img[:,:,2] == 255] = 254

    return img


def SplitImage(img, name, params):
    """
    Splits images into something close to 4000 pixels each direction
    saves the large image into 'unsplit' folder saves smaller images into
    image path
    TODO: Fix problem when one dimension is between 4000 and 7000
    TODO: Fill in the doc string better

    Parameters
    ----------
    img : array
        DESCRIPTION.
    name : string
        DESCRIPTION.
    params : dict
        DESCRIPTION.

    Returns
    -------
    nimg : array
        random selection of the splits

    """
    ydim, xdim = img.shape[:2]
    num_x_splits = img.shape[1] // 4000
    if num_x_splits == 0:
        num_x_splits = 1
    num_y_splits = img.shape[0] // 4000
    if num_y_splits == 0:
        num_y_splits = 1
    x_stride = xdim/num_x_splits
    y_stride = ydim/num_y_splits
    Z = MakeWindows(num_x_splits, num_y_splits, x_stride, y_stride)

    randint = random.randint(0, len(Z) - 1)
    for i in range(len(Z)):
        matplotlib.image.imsave(params['images_path'] + name + '_split_' + \
                                str(i) + '.png',
                                img[Z[i][0]:Z[i][1],Z[i][2]:Z[i][3],:])
        if i == randint:
            nimg = img[Z[i][0]:Z[i][1],Z[i][2]:Z[i][3],:].copy()
    os.rename(params['images_path'] + name + ".tif",
              params['unsplit_pat'] + name + ".tif")
    print(name, "Moved to unsplit directory")
    fimg = fip(nimg)
    return fimg


def SplitFlag(im_shape):
    """
    SplitFlag figures out what size of image it's dealing and sets an
      appropriate step size. over Some tif images are are over 30,000 pixels
      in one direction.
    Makes one of four choices over each dimension:
      If it is below 1000 pixels in that direction, it will return step size
      in that direction 2

      If it is between 1000 and 3000 it will return step size 4

      If it is between 3000 and 8000 it will return step size 6

      If it is over 8000 it will return step size 0 and return
      spf (split flag) = True
    TODO: fill in doc string better


    Parameters
    ----------
    im_shape : TYPE
        DESCRIPTION.

    Returns
    -------
    num_x_steps : TYPE
        DESCRIPTION.
    num_y_steps : TYPE
        DESCRIPTION.
    spf : TYPE
        DESCRIPTION.

    """
    spf = False
    print("im_shape=", im_shape)
    if im_shape[1] < 1000:
        num_x_steps = 2
    elif im_shape[1] < 3000:
        num_x_steps = 4
    elif im_shape[1] < 8000:
        num_x_steps = 6
    else:
        print("This image is too wide. It needs to split the image into smaller" +
              " images. This will be done automagically and the unsplit file will" +
              " be moved to the unsplit folder\n\n")
        spf = True
        num_x_steps = 0

    if im_shape[0] < 1000:
        num_y_steps = 2
    elif im_shape[0] < 3000:
        num_y_steps = 4
    elif im_shape[0] < 8000:
        num_y_steps = 6
    else:
        print("This image is too tall. It needs to split the image into smaller" +
              " images. This will be done automagically and the unsplit file will" +
              " be moved to the unsplit folder\n\n")
        spf = True
        num_y_steps = 0
    return num_x_steps, num_y_steps, spf


def MakeWindows(num_x_steps, num_y_steps, x_stride, y_stride):
    """
    Returns a matrix of x and y values to split the image at
    TODO: make this a more efficient lambda function
    TODO: fill in the doc string better

    Parameters
    ----------
    num_x_steps : TYPE
        DESCRIPTION.
    num_y_steps : TYPE
        DESCRIPTION.
    x_stride : TYPE
        DESCRIPTION.
    y_stride : TYPE
        DESCRIPTION.

    Returns
    -------
    Z : TYPE
        DESCRIPTION.

    """
    for i in range(num_y_steps):
        for j in range(num_x_steps):
            if i == 0 and j == 0:
                Z = np.array([0, np.int(y_stride), 0, np.int(x_stride)])
            else:
                Z = np.vstack((Z, [np.int(y_stride * i),
                                   np.int(y_stride * (i + 1)),
                                   np.int(x_stride * j),
                                   np.int(x_stride * (j + 1))]))
    return Z


def SlidingWindow(img, name, params):
    """
    Returns a matrix of
    # TODO: Make better docstring
    """
    num_x_steps, num_y_steps, spf = SplitFlag(img.shape)
    if spf:
        img = SplitImage(img, name, params)
        num_x_steps, num_y_steps, spf2 = SplitFlag(img.shape)
        new_name = name + '_split_0'
    ydim, xdim = img.shape[:2]
    x_stride = xdim/num_x_steps
    y_stride = ydim/num_y_steps
    Z = MakeWindows(num_x_steps, num_y_steps, x_stride, y_stride)
    if spf:
        return Z, img, new_name
    else:
        return Z, np.array([0]), name


def AnnoDraw(event, former_x, former_y, flags, param):
    global current_former_x, current_former_y, drawing, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_former_x, current_former_y = former_x, former_y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == True:
                cv2.line(im, (current_former_x,
                              current_former_y),
                         (former_x,former_y),
                         (0, 0, 255), param)
                current_former_x = former_x
                current_former_y = former_y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == True:
            cv2.line(im, (current_former_x, current_former_y),
                     (former_x, former_y),
                     (0, 0, 255), param)
            current_former_x = former_x
            current_former_y = former_y
    return former_x, former_y


def LabelWindow(img, Z, params, screen_size, im_mean):
    global drawing, mode, im
    print("Initial brush width = " + str(params['lw']))
    print("Change using the +/- keys")
    print("Cycle classes with [ESC]")
    print("Subtract mean with [Space]")
    print("Go back a frame with [b]")
    print("Skip a frame with [s]")
    lw = params['lw']
    drawing = False  # True if mouse is pressed
    mode = True  # if True, draw rectangle. Press 'm' to toggle to curve
    out = np.zeros((img.shape[0], img.shape[1]))

    ck = 0
    while ck < len(Z):
        ref_img = img.copy()
        if ck < 1:
            ck = 0
        im = img[Z[ck][0]:Z[ck][1],Z[ck][2]:Z[ck][3],:].copy()
        # FIX: See below
        im[im[:, :, 2] == 255] = 254
        cv2.rectangle(ref_img,
                      (Z[ck][2], Z[ck][0]),
                      (Z[ck][3], Z[ck][1]),
                      (255, 255, 0), 2)

        cv2.namedWindow('whole image', cv2.WINDOW_NORMAL)
        cv2.imshow('whole image', ref_img)
        cv2.resizeWindow('whole image', WinScales(screen_size, ref_img.shape))

        nav = False   # Navigator variable
        counter = 1   # Label number
        sm = 0        # Enhancement variable
        if np.std(im) > 0:
            Lc = np.zeros(im[:,:,0].shape)
            for label in params['classes'].keys():
                if nav:
                    print
                    continue
                imcopy = im.copy()
                cv2.namedWindow(label, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(label, WinScales(screen_size, ref_img.shape))
                cv2.moveWindow(label, 0, 0)  # Move it to (0,0)
                cv2.setMouseCallback(label, AnnoDraw, lw)
                while(1):
                    cv2.imshow(label, im)
                    k = cv2.waitKey(1)&0xFF
                    if k == 122:                          # If Z is pressed
                        im = imcopy.copy()
                        imcopy = im.copy()
                    if k == 27:                           # If ESC is pressed
                        # TODO: Find a better way to extract the drawing inputs
                        # Clouds often have a 255 so I made everything that
                        # was originally 255 in blue band == 254
                        try:
                            # This changes the section of the image that was
                            #  drawn on, works well, but if you want to go back
                            #  a label, I don't have a way to do that currently
                            Lc[im[:, :, 2] == 255] = counter
                            im[im[:, :, 2] == 255] = 160
                        except:
                            Lc[im == 255] = counter
                            im[im == 255] = 160
                        print(np.unique(Lc))
                        print(counter)

                        counter += 1
                        break

                    if k == 115:                         # If s is pressed
                        nav = True
                        break

                    if k == 98:                          # If b is pressed
                        nav = True
                        ck -= 2
                        break

                    if k == 43:                          # If + is pressed
                        lw += 1
                        print("brush width = " + str(lw))

                    if k == 45:                          # If - is pressed
                        lw -= 1
                        if lw < 1:
                            lw = 1
                        print("brush width = " + str(lw))
                    # TODO: This does not work well
                    if k == 32:                          # If SPACE is pressed
                        if sm == 0:
                            im = im - im_mean
                            im[im < 0] = 0
                            sm += 1
                        else:
                            im = imcopy.copy()
                            sm = 0
                cv2.destroyWindow(label)

#            try:
#                Lc = im[:, :, 2]
#            except:
#                Lc = im[:, :]

            if k != 98 and k != 115:
#                Lc[Lc > counter] = 0
                out[Z[ck][0]:Z[ck][1],Z[ck][2]:Z[ck][3]] = Lc
                print(np.unique(out), "  ", out)

        else:
#            bg_label = 0
#            if 'BackGround' in params['classes'].keys():
#                bg_label += int(list(params['classes'].keys()).index("BackGound"))
            try:
                Lc = np.zeros(np.shape(im[:,:,2])) + bg_label
            except:
                Lc = np.zeros(np.shape(im)) + bg_label

            out[Z[ck][0]:Z[ck][1],Z[ck][2]:Z[ck][3]] = Lc

        cv2.destroyWindow('whole image')
        ck += 1
    return out
