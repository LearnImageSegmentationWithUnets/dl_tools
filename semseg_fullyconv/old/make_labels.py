## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

import numpy as np
from scipy.io import loadmat
from scipy.misc import imsave

from collections import namedtuple
from glob import glob

Label = namedtuple('Label', ['name', 'color'])

label_defs = [
    Label('anthro',     (255,     0,   255)),
    Label('foam',       (255,  255,   255)),
    Label('terrain',        ( 102,   51,  0)),
    Label('water',          (0,  0, 255)),
    Label('veg',      (0,  255, 0)),
    Label('sand',       (255, 255, 0)),
    Label('road',    (255, 0, 0))]

files = glob('*.mat')

for file_in in files:
   dat = loadmat(file_in)

   out = np.zeros((np.shape(dat['class'])[0], np.shape(dat['class'])[1], 3), dtype='uint8')

   for k in np.unique(dat['class']):
      out[:,:,0][dat['class']==k] = label_defs[k].color[0]
      out[:,:,1][dat['class']==k] = label_defs[k].color[1]
      out[:,:,2][dat['class']==k] = label_defs[k].color[2]
   
   imsave(file_in.split('.TIF-0_ares.mat')[0]+'_gtFine_color.png', out)
   