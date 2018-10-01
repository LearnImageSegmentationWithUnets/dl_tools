## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

import numpy as np
from scipy.io import loadmat
from scipy.misc import imsave

from collections import namedtuple
from glob import glob
import sys

import s3fs
fs = s3fs.S3FileSystem(anon=True)

#if sys.version[0]=='3':
#   from tkinter import Tk
#   from tkinter.filedialog import askopenfilename
#else:
#   from Tkinter import Tk
#   from tkFileDialog import askopenfilename  

from skimage.filters.rank import median
from skimage.morphology import disk   

def rgb2bgr(tpl):
    return (tpl[2], tpl[1], tpl[0])

#==============================================================
if __name__ == '__main__':

   # read the WYSS section for how to run this
   script, direc = sys.argv        

   Label = namedtuple('Label', ['name', 'color'])

   files = [f for f in fs.ls(direc) if f.endswith('.mat')]

   with open('labeldefs.txt') as f:
      labels = f.readlines()
   labels = [x.strip() for x in labels]

   label_defs = []
   for label in labels:
      x,r,g,b = label.split(',')
      r = int(r.strip())
      g = int(g.strip())
      b = int(b.strip())   
      label_defs.append(Label(x,(r,g,b)))
   
   #Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing   
   #files = askopenfilename(filetypes=[("pick mat files","*.mat")], multiple=True)  

   for file_in in files:
      dat = loadmat(file_in)
      dat['class'] = median(dat['class'], disk(7))

      out = np.zeros((np.shape(dat['class'])[0], np.shape(dat['class'])[1], 3), dtype='uint8')

      for k in np.unique(dat['class']):
         out[:,:,0][dat['class']==k] = label_defs[k].color[0]
         out[:,:,1][dat['class']==k] = label_defs[k].color[1]
         out[:,:,2][dat['class']==k] = label_defs[k].color[2]
      imsave(file_in.split('_mres')[0]+'_gtFine_color.png', out)
   
   
   
   
   