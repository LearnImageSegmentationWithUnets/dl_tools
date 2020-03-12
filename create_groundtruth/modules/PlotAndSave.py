# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 21:33:43 2020

@author: Brodie
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def FlipAxForMPL(img):
    r = img[:,:,0]
    g = img[:,:,1]
    b = img[:,:,2]
    newim = np.dstack([b, g, r])
    return newim

def PlotAndSave(img, resr, crf_output, name, params):
    mpl_im = FlipAxForMPL(img.copy())

    if params['plot_c']:
        sp = 130
    else:
        sp = 120
    # ========================================================================
    print('Generating plot ....')
    cmap = colors.ListedColormap(list(params['classes'].values()))
    fig = plt.figure()
    fig.subplots_adjust(wspace=0.4)
    ax1 = fig.add_subplot(sp + 1)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    _ = ax1.imshow(mpl_im)
    plt.title(params['a_label'], loc='left', fontsize=params['font_size'])

    # ========================================================================
    if params['plot_c']:
        ax1 = fig.add_subplot(122)
        ax1.get_xaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)

        _ = ax1.imshow(mpl_im)
        plt.title(params['b_label'], loc='left', fontsize=params['font_size'])
        im2 = ax1.imshow(crf_output - 1,
                         cmap=cmap,
                         alpha=params['alpha_percent'],
                         vmin=0, vmax=len(params['classes']))
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%")
        cb = plt.colorbar(im2, cax=cax)
        cb.set_ticks(0.5 + np.arange(len(params['classes']) + 1))
        cb.ax.set_yticklabels(params['classes'])
        cb.ax.tick_params(labelsize=4)
    if params['plot_c']:
        ax1 = fig.add_subplot(133)
    else:
        ax1 = fig.add_subplot(122)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    _ = ax1.imshow(mpl_im)
    plt.title(params['c_label'],
              loc='left',
              fontsize=params['font_size'])
    im2 = ax1.imshow(resr,
                     cmap=cmap,
                     alpha=params['alpha_percent'],
                     vmin=0, vmax=len(params['classes']))
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%")
    cb=plt.colorbar(im2, cax=cax)
    cb.set_ticks(0.5 + np.arange(len(params['classes']) + 1))
    cb.ax.set_yticklabels(params['classes'])
    cb.ax.tick_params(labelsize=4)
    plt.savefig(params['compar_path'] + name + '_mres.png',
                dpi=600, bbox_inches = 'tight')
    del fig; plt.close()
    cv2.imwrite(params['result_path'] + name + '_mres_label.png',
                np.round(255*(resr/np.max(resr))).astype('uint8'))