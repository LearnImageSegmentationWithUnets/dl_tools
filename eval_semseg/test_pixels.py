## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

import os, sys, getopt
from scipy.io import savemat, loadmat
from glob import glob
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

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

#=====================================================
def get_stats(cm):

    m, n = np.shape(cm)
	
    TP = []
    for x in range(n):
        TP.append(cm[x, x])
    
    FP = []
    for x in range(n):
        FP.append(sum(cm[:, x])-cm[x, x])

    FN = []
    for x in range(n):
        FN.append(sum(cm[x, :], 2)-cm[x, x])    
        
    tp = np.asarray(TP)
    fp = np.asarray(FP)
    fn = np.asarray(FN)    
        
    p = tp/(tp+fp)
    p[p==0] = np.nan

    r = tp/(tp+fn)
    r[r==0] = np.nan    

    p = np.nanmean(p)
    r = np.nanmean(r)
    
    f = 2*((p*r)/(p+r))
    
    return p, r, f
	
## =========================================================
def plot_confusion_matrix2(cm, classes, normalize=False, cmap=plt.cm.Blues, dolabels=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm[np.isnan(cm)] = 0

    plt.imshow(cm, interpolation='nearest', cmap=cmap, vmax=1, vmin=0)
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    if dolabels==True:
       tick_marks = np.arange(len(classes))
       plt.xticks(tick_marks, classes, fontsize=3) # rotation=45
       plt.yticks(tick_marks, classes, fontsize=3)

       plt.ylabel('True label',fontsize=6)
       plt.xlabel('Estimated label',fontsize=6)

    else:
       plt.axis('off')

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if cm[i, j]>0:
           plt.text(j, i, format(cm[i, j], fmt),
                 fontsize=3,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()

    return cm


#==============================================================
if __name__ == '__main__':

   #direc = ''; tile = ''

   #argv = sys.argv[1:]
   #try:
   #   opts, args = getopt.getopt(argv,"ht:")
   #except getopt.GetoptError:
   #   print('python test_class_tiles.py -t tile')
   #   sys.exit(2)

   #for opt, arg in opts:
   #   if opt == '-h':
   #      print('Example usage: python test_class_tiles.py -t tile')
   #      sys.exit()
   #   elif opt in ("-t"):
   #      tile = arg	

   #===============================================
   # Run main application
   Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing   
   
   direc = askdirectory()	
	
   ##tile = 192 #96 #128 #160 #192 #224
   ##direc='test'

   ##tile = int(tile)
   
   mres = sorted(glob(direc+os.sep+'*mres*.mat'))

   ares = sorted(glob(direc+os.sep+'*ares*.mat'))
   

   for k in range(len(ares)):
      print("loading "+ares[k])
      print("loading "+mres[k])
	  
      a = loadmat(ares[k])['class']
      c = loadmat(mres[k])['class']

      alabs = loadmat(ares[k])['labels']
      clabs = loadmat(mres[k])['labels']
      alabs = [label.replace(' ','') for label in alabs]
      clabs = [label.replace(' ','') for label in clabs]
      cind = [clabs.index(x) for x in alabs]
      aind = [alabs.index(x) for x in alabs]

      if k==0:
         Cmaster = np.zeros((len(alabs), len(alabs)))

      c2 = c.copy()
      for kk in range(len(aind)):
         if cind[kk] != aind[kk]:
            c2[c==cind[kk]] = aind[kk] 
      del c

      #e = precision_recall_fscore_support(a.flatten(), c2.flatten())
      #p = np.mean(e[0])
      #r = np.mean(e[1])
      #f = np.mean(e[2])
   
      CM = confusion_matrix(a.flatten(), c2.flatten())

      CM = np.asarray(CM)
	  
      p, r, f = get_stats(CM)	  	  
      print('precision: %f' %(p))
      print('recall: %f' %(r))
      print('f-score: %f' %(f))
   
      fig = plt.figure()
      ax1 = fig.add_subplot(221)
      plot_confusion_matrix2(CM, classes=alabs, normalize=True, cmap=plt.cm.Reds)
      plt.savefig(ares[k].split(os.sep)[-1].split('.mat')[0]+'cm.png', dpi=300, bbox_inches='tight')
      del fig; plt.close()
	  	  

      #Cmaster += CM

   #p, r, f = get_stats(Cmaster)	  	  
   #print('precision: %f' %(p))
   #print('recall: %f' %(r))
   #print('f-score: %f' %(f))


