
## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

# A small utility to move all provided 'gt' files into the folder (test or train) that contains the associated image

from glob import glob
import shutil, os

testfiles = glob('..'+os.sep+'demo_data'+os.sep+'test'+os.sep+'*.*')

trainfiles = glob('..'+os.sep+'demo_data'+os.sep+'train'+os.sep+'*.*')

matfiles = glob('..'+os.sep+'demo_data'+os.sep+'gt'+os.sep+'*.*')

for f in matfiles:
   name = f.split(os.sep)[-1].split('_mres.mat')[0] 
   
   matchfile = [file for file in trainfiles if file.split(os.sep)[-1].startswith(name)]
   if matchfile:
      shutil.copy(f, '..'+os.sep+'demo_data'+os.sep+'train')	
	  
   matchfile = [file for file in testfiles if file.split(os.sep)[-1].startswith(name)]
   if matchfile:
      shutil.copy(f, '..'+os.sep+'demo_data'+os.sep+'test')		  