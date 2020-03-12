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
import rasterio
from skimage.filters import threshold_otsu
from skimage.exposure import rescale_intensity

class MaskPainter():
    def __init__(self, image, params, screen_size, im_mean):
        self.image = image
        self.params = params
        self.screen_size = screen_size
        self.im_mean = im_mean
        self.class_mask = np.zeros((self.image.shape[0],
                                   self.image.shape[1],
                                   int(len(self.params['classes']) + 1)))
        self.mask_copy = self.class_mask.copy()
        self.size = self.params['lw']
        self.current_x = 0
        self.current_y = 0

    def WinScales(self, imshape):
        dim = None
        (height, width) = self.screen_size
        (h, w) = imshape

        if width > w and height > h:
            return imshape
        else:
            rh = float(height) / float(h)
            rw = float(width) / float(w)
            rf = min(rh, rw) * self.params['ref_im_scale']
            dim = (int(h * rf), int(w * rf))
        return dim

    def MakeWindows(self):
        """
        Returns a matrix of x and y values to split the image at

        Returns
        -------
        Z : array
            x and y coordinates (x_min, y_min, x_max, y_max)
            of the sections of the whole image to be labeled
        """
        num_x_steps, num_y_steps = StepCalc(self.image.shape,
                                            self.params['max_x_steps'],
                                            self.params['max_y_steps'])
        ydim, xdim = self.image.shape[:2]
        x_stride = xdim/num_x_steps
        y_stride = ydim/num_y_steps
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
    
    def Enhance(self, img, sm):
        b = img[:, :, 0]
        g = img[:, :, 1]
        r = img[:, :, 2]
        def Rescale(r, g, b):
            # TODO: How do I loop over third axis?
            # for ii, band in enumerate(arr):
            r_min = np.min(r)
            g_min = np.min(g)
            b_min = np.min(b)
            r_max = np.max(r)
            g_max = np.max(g)
            b_max = np.max(b)
            r_rescaled = rescale_intensity(r, in_range=(r_min, r_max),
                                           out_range=(0, 256))
            g_rescaled = rescale_intensity(g, in_range=(g_min, g_max),
                                           out_range=(0, 256))
            b_rescaled = rescale_intensity(b, in_range=(b_min, b_max),
                                           out_range=(0, 256))
            return r_rescaled, g_rescaled, b_rescaled

        def GaussianBl(im, size):
            """
            Trying out Gausin blur technique I got from the drone pic processing
            company ???
            """
            im = im/255
            gaussian_rgb = cv2.GaussianBlur(im, size, 10.0)
            gaussian_rgb[gaussian_rgb < 0] = 0
            gaussian_rgb[gaussian_rgb > 1] = 1
            unsharp_rgb = cv2.addWeighted(im, 1.5, gaussian_rgb, -0.5, 0)
            unsharp_rgb[unsharp_rgb < 0] = 0
            unsharp_rgb[unsharp_rgb > 1] = 1
            return unsharp_rgb

        if sm == 1:
            # Do the NDWI-like
            ndwi = ((b - r) / (b + r)).astype('uint8')
            b = ndwi
            g = ndwi
            r = ndwi
            r, g, b = Rescale(r, g, b)
            ndwi_im = np.dstack((b, g, r))
            ndwi_im[ndwi_im[:, :, 2] == 255] = 254
            return ndwi_im

        elif sm == 2:
            # Shadow index
            si = (np.sqrt((256 - b) * (256 - g))).astype('uint8')
            b = si
            g = si
            r = si
            r, g, b = Rescale(r, g, b)
            si_im = np.dstack((b, g, r))
            si_im[si_im[:, :, 2] == 255] = 254
            return si_im

        elif sm == 3:
            img_blur = GaussianBl(img, (5, 5))
            return img_blur

        elif sm == 4:
            img_blur = GaussianBl(img, (5, 5))
            img_gray = np.mean(img_blur, axis=-1)
            p_low, p_high = np.percentile(img_gray, (1, 95))
            img_gray = rescale_intensity(img_gray, in_range=(p_low, p_high))
            img_bin = img_gray > threshold_otsu(img_gray)
            img_edge = np.mean(img_bin, axis=1)
            b = img_edge
            g = img_edge
            r = img_edge
            r, g, b = Rescale(r, g, b)
            othu = np.dstack((b, g, r))
            othu[othu[:, :, 2] == 255] = 254
            return othu

        else:
            return img

    def AnnoDraw(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.draw = True
            self.current_x, self.current_y = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.draw:
                cv2.line(self.im_sect, (self.current_x,
                                        self.current_y),
                         (x, y),
                         (0, 0, 255), param)
                self.current_x = x
                self.current_y = y

        elif event == cv2.EVENT_LBUTTONUP:
            self.draw = False

        return x, y

    def Overlay(self, src, overlay):
        #                 (src , overlay):
        """
        Returns a new image to display, after blending in the pixels that have
            been labeled

        Takes
        src : array
            Original image
        overlay : array
            Overlay image, in this case the Labels

        Returns
        -------
        new_im : array
            Blended im
        """
        if np.max(overlay) > 0:
            new_im = src.copy()
            vals = np.argmax(overlay, axis=2)
            vals *= 18
            new_im[:, :, 0][vals > 0] = vals[vals > 0]
            return new_im
        else:
            return src

    def LabelWindow(self):
        print("Initial brush width = 5")
        print("  -change using the +/- keys")
        print("Cycle classes with [ESC]")
        print("Subtract mean with [Space]")
        print("Go back a frame with [b]")
        print("Skip a frame with [s]")
        print("\nTo navigate labels use:\nButton: Label")
        nav_b = "123456789qwerty"
        for labl, button in enumerate(self.params['classes'].keys()):
            print(button + ':', nav_b[labl])
        nav_b = nav_b[:len(self.params['classes'])]
        self.draw = False  # True if mouse is pressed
        self.Z = self.MakeWindows()
        lab = False
        ck = 0
        while ck < len(self.Z):
            ref_img = self.image.copy()
            if ck < 1:
                ck = 0
            self.im_sect = self.image[self.Z[ck][0]:self.Z[ck][1],
                                      self.Z[ck][2]:self.Z[ck][3],
                                      :].copy()
            # FIX: See below
            self.im_sect[self.im_sect[:, :, 2] == 255] = 254
            cv2.rectangle(ref_img,
                          (self.Z[ck][2], self.Z[ck][0]),
                          (self.Z[ck][3], self.Z[ck][1]),
                          (255, 255, 0), 2)

            cv2.namedWindow('whole image', cv2.WINDOW_NORMAL)
            cv2.imshow('whole image', ref_img)
            cv2.resizeWindow('whole image',
                             (self.WinScales(ref_img.shape[:2])))
            cv2.moveWindow('whole image', 0, 28)
            nav = False   # Navigator variable
            if not lab:
                counter = 1   # Label number
            sm = 0        # Enhancement variable
            if np.std(self.im_sect) > 0:
                s = np.shape(self.im_sect[:, :, 2])
                if not lab:
                    # TODO: Lc should never be set to zeros!!
                    #      It needs to get from class mask, so that it can
                    #      keep the labels that have been done
                    #      Need to figure out what that means for lab and nav
                    Lc = np.zeros((s[0], s[1],
                                   len(self.params['classes']) + 1))
                else:
                    Lc[counter] = Lc[counter] * 0
                while counter <= len(self.params['classes']):
                    label = list(self.params['classes'].keys())[counter - 1]
                    if nav:
                        break
                    imcopy = self.im_sect.copy()
                    cv2.namedWindow(label, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(label,
                                     tuple(self.WinScales(imcopy.shape[:2])))
                    cv2.moveWindow(label, 0, 28)  # Move it to (0,0)
                    cv2.setMouseCallback(label, self.AnnoDraw, self.size)
                    while(1):
                        showim = self.Overlay(self.im_sect, Lc)
                        cv2.imshow(label, showim)
                        k = cv2.waitKey(1) & 0xFF
                        if k == ord("z"):     # If Z is pressed
                            self.im_sect = imcopy.copy()
                            imcopy = self.im_sect.copy()
                        if k == 27:  # If ESC is pressed, copy labeled pixels
                            #           to Lc and go to next label
                            # TODO: Find a better way to extract the drawing
                            #       inputs Clouds often have a 255 so I made
                            #       everything that was originally 255 in
                            #       blue band == 254
                            try:
                                # This changes the section of the image that
                                #   was drawn on, works well, but if you want
                                #   to go back a label, I don't have a way to
                                #   do that currently
                                Lc[:,
                                   :,
                                   counter][self.im_sect[:,
                                                         :,
                                                         2] == 255] = counter
                                self.im_sect[self.im_sect[:,
                                                          :,
                                                          2] == 255] = 160
                            except:
                                Lc[:, :, counter][self.im_sect == 255] = \
                                    counter
                                self.im_sect[self.im_sect == 255] = 160
                            counter += 1
                            break

                        if chr(k) in nav_b:  # if label number pressd, go to it
                            nav = True
                            lab = True
                            ck -= 1
                            counter = nav_b.find(chr(k)) + 1
                            break

                        if k == ord("s"):  # If s is pressed, skip square
                            nav = True
                            break

                        if k == ord('b'):  # If b is pressed go back a square
                            nav = True
                            ck -= 2
                            break

                        if k == ord('-'):  # If + is pressed, increase brush wi
                            self.size += 1
                            print("brush width = " + str(self.size))

                        if k == ord('-'):  # If - is pressed, decrese brush wid
                            self.size -= 1
                            if self.size < 1:
                                self.size = 1
                            print("brush width = " + str(self.size))
                        # TODO: This does not work well, need to have a couple
                        #       of enhancement options and a way to bring back
                        #       the original image with labels
                        if k == ord(' '):  # If SPACE is pressed, do enhancement
                            if sm == 0:
                                self.im_sect = self.im_sect - self.im_mean
                                self.im_sect[self.im_sect < 0] = 0
                                sm += 1
                            elif sm < 4:
                                self.im_sect = self.Enhance(imcopy.copy(), sm)
                                sm += 1
                            elif sm == 3:
                                thresh = threshold_otsu(imcopy.copy())
                                self.im_sect = self.im_sect > thresh
                                sm += 1
                            else:
                                self.im_sect = imcopy.copy()
                                sm = 0
                    cv2.destroyWindow(label)

                if not nav:
                    self.class_mask[self.Z[ck][0]:self.Z[ck][1],
                                    self.Z[ck][2]:self.Z[ck][3], :] = Lc
                    lab = False
            else:
                if self.params['auto_class'] == "No":
                    ac = 0
                else:
                    ac = list(self.params['classes'].keys()).index('BackGrnd') + 1
                self.class_mask[self.Z[ck][0]:self.Z[ck][1],
                                self.Z[ck][2]:self.Z[ck][3], :] = \
                    np.ones((self.im_sect.shape[0],
                            self.im_sect.shape[1],
                            self.class_mask.shape[2])) * ac

            cv2.destroyWindow('whole image')
            ck += 1
        return np.argmax(self.class_mask, axis=2)


def MakeWindows(image):
    """
    #TODO: Delete this and use the one in MaskPainter!!!!
    I don't know how to call the correct from below yet
    """
    num_x_steps, num_y_steps = StepCalc(image.shape)
    ydim, xdim = image.shape[:2]
    x_stride = xdim/num_x_steps
    y_stride = ydim/num_y_steps
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

def ReadGeotiff(image_path, rgb):
    """
    This function reads image in GeoTIFF format.
    TODO: Fill in the doc string better

    Parameters
    ----------
    image_path : string
        full or relative path to the tiff image
    rgb : TYPE
        is it RGB or BGR

    Returns
    -------
    img : array
        2D or 3D numpy array of the image
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


def SplitImage(img, name, ext, params):
    """
    Splits images into something close to 4000 pixels each direction
    saves the large image into 'unsplit' folder saves smaller images into
    image path
    TODO: Fix problem when one dimension is between 4000 and 7000

    Parameters
    ----------
    img : array
        Image
    name : string
        Image name
    ext : string
        Extension string, so it can save the original file to the unsplit
        directory
    params : dict
        Parameters defined in the parameters script

    Returns
    -------
    Complete : Bool
        Just indicates that it was successful
    """
    Z = MakeWindows(img)
    for i in range(len(Z)):
        matplotlib.image.imsave(params['images_path'] + name + '_split_' + \
                                str(i) + '.png',
                                img[Z[i][0]:Z[i][1],Z[i][2]:Z[i][3],:])
    os.rename(params['images_path'] + name + ext,
              params['unsplit_pat'] + name + ext)
    print(name, "Moved to unsplit directory")
    return True


def StepCalc(im_shape, max_x_steps=None, max_y_steps=None):
    """
    SplitFlag figures out what size of image it's dealing and sets an
      appropriate step size. Some tif images are are over 30,000 pixels
      in one direction.

    Parameters
    ----------
    im_shape : TYPE
        Tuple of image shape

    Returns
    -------
    num_x_steps : TYPE
        how many windows are needed in the x direction
    num_y_steps : TYPE
        how many windows are needed in the x direction
    """
    if max_x_steps == None:
        if im_shape[1] < 1000:
            num_x_steps = 2
        elif im_shape[1] < 2000:
            num_x_steps = 3
        elif im_shape[1] < 3000:
            num_x_steps = 4
        elif im_shape[1] < 5000:
            num_x_steps = 5
        else:
            num_x_steps = 6
    else:
        num_x_steps = max_x_steps

    if max_y_steps == None:
        if im_shape[0] < 1000:
            num_y_steps = 2
        elif im_shape[0] < 2000:
            num_y_steps = 3
        elif im_shape[0] < 3000:
            num_y_steps = 4
        elif im_shape[0] < 5000:
            num_y_steps = 5
        else:
            num_y_steps = 6
    else:
        num_y_steps = max_y_steps

    return num_x_steps, num_y_steps


def SplitCheck(im_shape):
    if im_shape[0] > 7000 or im_shape[1] > 7000:
        return True
    else:
        return False