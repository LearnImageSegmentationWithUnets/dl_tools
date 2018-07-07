## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

## This function will take an directory of images and sort them into training and testing sets, based on a given split

from imageio import imread, imsave
import sys, getopt, os, shutil, glob
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

   #===============================================
   # Run main application
   Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing   
   direc = askdirectory()   
   direc = os.path.normpath(direc)
   
   #=======================================================
   try:
      os.mkdir(os.path.dirname(direc)+os.sep+'train')
   except:
      print('train directory could not be made - check inputs')
      sys.exit(2)
	  
   try:
      os.mkdir(os.path.dirname(direc)+os.sep+'test')
   except:
      print('test directory could not be made - check inputs')
      sys.exit(2)

   allfiles = glob.glob(direc+os.sep+'*.*')	
   print(str(len(allfiles))+' files found')   
   trainfiles = np.random.choice(allfiles, int(prop_train*len(allfiles)), replace=False)	  
   testfiles = [x for x in allfiles if x not in trainfiles.tolist()]  
   print('moving '+str(len(trainfiles))+' to train')      
   for f in trainfiles:
      shutil.copy(f, os.path.dirname(direc)+os.sep+'train')	
   print('moving '+str(len(testfiles))+' to test')      		 
   for f in testfiles:
      shutil.copy(f, os.path.dirname(direc)+os.sep+'test')		  

	  
	  
