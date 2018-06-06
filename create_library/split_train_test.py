## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

from imageio import imread, imsave
import sys, getopt, os, shutil, glob
from joblib import Parallel, delayed
import numpy as np

if sys.version[0]=='3':
   from tkinter import Tk, Toplevel 
   from tkinter.filedialog import askopenfilename, askdirectory
   import tkinter
   import tkinter as tk
   from tkinter.messagebox import *   
   from tkinter.filedialog import *
else:
   from Tkinter import Tk, TopLevel
   from tkFileDialog import askopenfilename, askdirectory
   import Tkinter as tkinter
   import Tkinter as tk
   from Tkinter.messagebox import *   
   from Tkinter.filedialog import *   

   
#==============================================================
def cp_files(direc, label, prop_train):
   allfiles = glob.glob(direc+os.sep+label+os.sep+'*.*')	
   print(str(len(allfiles))+' files found in: '+label)   
   trainfiles = np.random.choice(allfiles, int(prop_train*len(allfiles)), replace=False)	  
   testfiles = [x for x in allfiles if x not in trainfiles.tolist()]  
   for f in trainfiles:
      shutil.copy(f, os.path.dirname(direc)+os.sep+'train'+os.sep+label)	
		 
   for f in testfiles:
      shutil.copy(f, os.path.dirname(direc)+os.sep+'test'+os.sep+label)		  
	  
#==============================================================
if __name__ == '__main__':

   argv = sys.argv[1:]
   try:
      opts, args = getopt.getopt(argv,"hp:")
   except getopt.GetoptError:
      print('python split_train_test.py -p 0.5')
      sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print('Example usage: split_train_test.py -p 0.5')
         sys.exit()
      elif opt in ("-p"):
         prop_train = arg
		 
   prop_train = float(prop_train)		 
   
   direc = os.path.normpath(direc)
   labels = [os.path.basename(x[0]) for x in os.walk(direc)][1:]
   
   #=======================================================
   try:
      os.mkdir(os.path.dirname(direc)+os.sep+'train')
      for label in labels:
         os.mkdir(os.path.dirname(direc)+os.sep+'train'+os.sep+label)
   except:
      print('train directory could not be made - check inputs')
      sys.exit(2)
	  
   try:
      os.mkdir(os.path.dirname(direc)+os.sep+'test')
      for label in labels:
         os.mkdir(os.path.dirname(direc)+os.sep+'test'+os.sep+label)
   except:
      print('test directory could not be made - check inputs')
      sys.exit(2)
	  
   w = Parallel(n_jobs=-1, verbose=0)(delayed(cp_files)(direc, label, prop_train) for label in labels)

	  
	  
