## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

#general
from __future__ import division
from joblib import Parallel, delayed
from glob import glob
import numpy as np 
from scipy.misc import imread
from scipy.io import loadmat
import sys, getopt, os

from tile_utils import *

from scipy.stats import mode as md
from scipy.misc import imsave

if sys.version[0]=='3':
   from tkinter import Tk, Toplevel 
   from tkinter.filedialog import askopenfilename
   import tkinter
   import tkinter as tk
   from tkinter.messagebox import *   
   from tkinter.filedialog import *
else:
   from Tkinter import Tk, TopLevel
   from tkFileDialog import askopenfilename
   import Tkinter as tkinter
   import Tkinter as tk
   from Tkinter.messagebox import *   
   from Tkinter.filedialog import *   
   
import os.path as path


# =========================================================
def writeout(tmp, cl, labels, outpath, thres):

   l, cnt = md(cl.flatten())
   l = np.squeeze(l)
   if cnt/len(cl.flatten()) > thres:
      outfile = id_generator()+'.jpg'
      fp = outpath+os.sep+labels[l]+os.sep+outfile
      imsave(fp, tmp)

#==============================================================
if __name__ == '__main__':

   direc = ''; tile = ''; thres = ''

   argv = sys.argv[1:]
   try:
      opts, args = getopt.getopt(argv,"ht:a:b:")
   except getopt.GetoptError:
      print('python retile.py -t tilesize -a threshold -b proportion_thin')
      sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print('Example usage: python retile.py -t 96 -a 0.9 -b 0.5')
         sys.exit()
      elif opt in ("-t"):
         tile = arg
      elif opt in ("-a"):
         thres = arg
      elif opt in ("-b"):
         thin = arg
		 
   if not direc:
      direc = 'train'
   if not tile:
      tile = 96
   if not thres:
      thres = .9
   if not thin:
      thin = 0
	  
   tile = int(tile)
   thres = float(thres)
   thin = float(thin)

   #===============================================
   # Run main application
   Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing   
   files = askopenfilename(filetypes=[("pick mat files","*.mat")], multiple=True)  
    
   direc = imdirec = os.path.dirname(files[0])##'useimages'
   
   #=======================================================
   outpath = direc+os.sep+'tile_'+str(tile)
   ##files = sorted(glob(direc+os.sep+'*.mat'))

   labels = loadmat(files[0])['labels']

   labels = [label.replace(' ','') for label in labels]
   #=======================================================

   #=======================================================
   try:
      os.mkdir(outpath)
   except:
      pass

   for f in labels:
      try:
         os.mkdir(outpath+os.sep+f)
      except:
         pass
   #=======================================================

   types = (direc+os.sep+'*.jpg', direc+os.sep+'*.jpeg', direc+os.sep+'*.tif', direc+os.sep+'*.tiff', direc+os.sep+'*.png') # the tuple of file types
   files_grabbed = []
   for f in types:
      files_grabbed.extend(glob(f))	   
   
   #=======================================================
   for f in files:

      dat = loadmat(f)
      labels = dat['labels']

      labels = [label.replace(' ','') for label in labels]	  	  
	  
      res = dat['class']
      del dat
      core = f.split('/')[-1].split('_mres')[0]  

      ##get the file that matches the above pattern but doesn't contain 'mres'	  
      fim = [e for e in files_grabbed if e.find(core)!=-1 if e.find('mres')==-1 ]
      if fim:
         fim = fim[0]
         print('Generating tiles from dense class map ....')
         Z,ind = sliding_window(imread(fim), (tile,tile,3), (int(tile/2), int(tile/2),3)) 

         C,ind = sliding_window(res, (tile,tile), (int(tile/2), int(tile/2))) 

         w = Parallel(n_jobs=-1, verbose=0, pre_dispatch='2 * n_jobs', max_nbytes=None)(delayed(writeout)(Z[k], C[k], labels, outpath, thres) for k in range(len(Z))) 
      else:
         print("correspodning image not found")	     
		 
   print('thinning files ...')
   if thin>0:
      for f in labels:   
         files = glob(outpath+os.sep+f+os.sep+'*.jpg')
         if len(files)>60:   
            usefiles = np.random.choice(files, int(thin*len(files)), replace=False)   
            rmfiles = [x for x in files if x not in usefiles.tolist()] 
            for rf in rmfiles:
               os.remove(rf)
	  
   for f in labels:
      files = glob(outpath+os.sep+f+os.sep+'*.jpg')
      print(f+': '+str(len(files)))
  

   

