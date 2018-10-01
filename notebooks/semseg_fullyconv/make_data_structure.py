## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

import sys, os, shutil
from glob import glob
from imageio import imread, imwrite
#from joblib import Parallel, delayed

import s3fs
fs = s3fs.S3FileSystem(anon=True)

#if sys.version[0]=='3':
#  from tkinter import Tk
#   from tkinter.filedialog import askdirectory
#else:
#   from Tkinter import Tk
#   from tkFileDialog import askdirectory 
  
def write_png(k, root, subset):
   k = k.split('_mres')[0]+'.jpg'
   kout = k.replace(os.sep+'gt','')
   koutpng = kout.replace('.jpg', '_RGB.png')
   koutpng = koutpng.split(os.sep)[-1]     
   
   imwrite('data'+os.sep+'samples'+os.sep+'RGB'+os.sep+subset+os.sep+'data'+os.sep+koutpng, imread(kout))   
   ##shutil.copy(k.replace(os.sep+'gt',''), root+os.sep+'data'+os.sep+'samples'+os.sep+'RGB'+os.sep+'train'+os.sep+'data')
   
#=============================================

#==============================================================
if __name__ == '__main__':

   # read the WYSS section for how to run this
   script, root = sys.argv        
    
   ## get top level directory
   #Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing   
   #root = askdirectory()  

   ###set up file structure for samples
   try:
      os.mkdir('data') ##root+os.sep+
      os.mkdir('data'+os.sep+'samples')  
      os.mkdir('data'+os.sep+'samples'+os.sep+'RGB')
      os.mkdir('data'+os.sep+'samples'+os.sep+'RGB'+os.sep+'train')
      os.mkdir('data'+os.sep+'samples'+os.sep+'RGB'+os.sep+'val')
      os.mkdir('data'+os.sep+'samples'+os.sep+'RGB'+os.sep+'train'+os.sep+'data')
      os.mkdir('data'+os.sep+'samples'+os.sep+'RGB'+os.sep+'val'+os.sep+'data')
   except:
      pass

   #get all mat files in test\gt	  
   valroot = root+os.sep+'test'+os.sep+'gt'+os.sep	     
   valfiles = [f for f in fs.ls(valroot) if f.endswith('.mat')] ##glob(valroot+'*.mat')
   #find all associated jpg files, convert to png, and write to new file structure  
   ##Parallel(n_jobs=-1, verbose=0)(delayed(write_png)(k, root, 'val') for k in valfiles) 

   for k in valfiles:
      write_png(k, root, 'val') 
    
   #get all mat files in train\gt	     
   trainroot = root+os.sep+'train'+os.sep+'gt'+os.sep   
   trainfiles = [f for f in fs.ls(trainroot) if f.endswith('.mat')] ##glob(trainroot+'*.mat')
   #find all associated jpg files, convert to png, and write to new file structure  
   ##Parallel(n_jobs=-1, verbose=0)(delayed(write_png)(k, root, 'train') for k in trainfiles) 

   for k in trainfiles:
      write_png(k, root, 'train') 

   ###set up file structure for labels
   try:
      os.mkdir(root+os.sep+'data'+os.sep+'labels')
      os.mkdir(root+os.sep+'data'+os.sep+'labels'+os.sep+'gtFine')
      os.mkdir(root+os.sep+'data'+os.sep+'labels'+os.sep+'gtFine'+os.sep+'train')
      os.mkdir(root+os.sep+'data'+os.sep+'labels'+os.sep+'gtFine'+os.sep+'val')
      os.mkdir(root+os.sep+'data'+os.sep+'labels'+os.sep+'gtFine'+os.sep+'train'+os.sep+'data')
      os.mkdir(root+os.sep+'data'+os.sep+'labels'+os.sep+'gtFine'+os.sep+'val'+os.sep+'data') 
   except:
      pass

   #trainfiles = glob(trainroot+'*.mat')
   for k in trainfiles:
      shutil.copy(k, root+os.sep+'data'+os.sep+'labels'+os.sep+'gtFine'+os.sep+'train'+os.sep+'data')
   
   
   valfiles = glob(valroot+'*.mat')
   for k in valfiles:
      shutil.copy(k, root+os.sep+'data'+os.sep+'labels'+os.sep+'gtFine'+os.sep+'val'+os.sep+'data')
   


     # valfiles = [glob(e) for e in [valroot+'*.jpg', valroot+'*.jpeg', valroot+'*.png', valroot+'*.tif', valroot+'*.tiff']]
   # valfiles = valfiles[np.argmax([len(x) for x in valfiles])]

   # for k in valfiles:
      # k = k.split('_mres')[0]+'.jpg'
      # kout = k.replace(os.sep+'gt','')
      # imsave(kout.replace('.jpg', '.png'), imread(kout))
      # ##shutil.copy(k.replace(os.sep+'gt',''), root+os.sep+'data'+os.sep+'samples'+os.sep+'RGB'+os.sep+'val'+os.sep+'data')
   


