## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

#python script to carry out quick semi-supervised segmentation of a single image
#uses manual labeling of regions of image as unary potentials for a fully-connected conditional random field

from __future__ import division

import sys, getopt, os
import time, socket 

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

import cv2
import numpy as np
from scipy.misc import imsave, imread, imresize 
from scipy.io import savemat

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from numpy.lib.stride_tricks import as_strided as ast
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels, unary_from_softmax

import os.path as path
from skimage.filters.rank import median
from skimage.morphology import disk

import rasterio
from osgeo import gdal,ogr,osr

## conda install rasterio gdal


##-------------------------------------------------------------
def read_geotiff(image_path): 
   """
   This function reads image in GeoTIFF format.
   """

   ##print('Reading GeoTIFF data ...')
   if type(image_path) is not list:
      input = [image_path]

   ## read all arrays
   bs = []
   for layer in input:
      with rasterio.open(layer) as src:
         layer = src.read()#[0,:,:]
      w, h = (src.width, src.height)
      xmin, ymin, xmax, ymax = src.bounds
      crs = src.get_crs()
      del src
      bs.append({'bs':layer, 'w':w, 'h':h, 'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax, 'crs':crs})

   ## resize arrays so common grid
   ##get bounds
   xmax = max([x['xmax'] for x in bs])
   xmin = min([x['xmin'] for x in bs])
   ymax = max([x['ymax'] for x in bs])
   ymin = min([x['ymin'] for x in bs])
   nz, nx, ny = np.shape(bs[0]['bs'])
   for k in range(len(bs)):
      bs[k]['h'] = nx
      bs[k]['w'] = ny
      bs[k]['xmin'] = xmin
      bs[k]['xmax'] = xmax
      bs[k]['ymin'] = ymin
      bs[k]['ymax'] = ymax

   img = np.dstack(bs[0]['bs']).astype('uint8').reshape((nx,ny,nz))

   return np.squeeze(img), bs
   
# =========================================================
def norm_shape(shap):
   '''
   Normalize numpy array shapes so they're always expressed as a tuple,
   even for one-dimensional shapes.
   '''
   try:
      i = int(shap)
      return (i,)
   except TypeError:
      # shape was not a number
      pass

   try:
      t = tuple(shap)
      return t
   except TypeError:
      # shape was not iterable
      pass

   raise TypeError('shape must be an int, or a tuple of ints')


# =========================================================
# Return a sliding window over a in any number of dimensions
# version with no memory mapping
def sliding_window(a,ws,ss = None,flatten = True):
    '''
    Return a sliding window over a in any number of dimensions
    '''
    if None is ss:
        # ss was not provided. the windows will not overlap in any direction.
        ss = ws
    ws = norm_shape(ws)
    ss = norm_shape(ss)
    # convert ws, ss, and a.shape to numpy arrays
    ws = np.array(ws)
    ss = np.array(ss)
    shap = np.array(a.shape)
    # ensure that ws, ss, and a.shape all have the same number of dimensions
    ls = [len(shap),len(ws),len(ss)]
    if 1 != len(set(ls)):
        raise ValueError(\
        'a.shape, ws and ss must all have the same length. They were %s' % str(ls))

    # ensure that ws is smaller than a in every dimension
    if np.any(ws > shap):
        raise ValueError(\
        'ws cannot be larger than a in any dimension.\
 a.shape was %s and ws was %s' % (str(a.shape),str(ws)))
    # how many slices will there be in each dimension?
    newshape = norm_shape(((shap - ws) // ss) + 1)
    # the shape of the strided array will be the number of slices in each dimension
    # plus the shape of the window (tuple addition)
    newshape += norm_shape(ws)
    # the strides tuple will be the array's strides multiplied by step size, plus
    # the array's strides (tuple addition)
    newstrides = norm_shape(np.array(a.strides) * ss) + a.strides
    a = ast(a,shape = newshape,strides = newstrides)
    if not flatten:
        return a
    # Collapse strided so that it has one more dimension than the window.  I.e.,
    # the new array is a flat list of slices.
    meat = len(ws) if ws.shape else 0
    firstdim = (np.product(newshape[:-meat]),) if ws.shape else ()
    dim = firstdim + (newshape[-meat:])
    # remove any dimensions with size 1
    #dim = filter(lambda i : i != 1,dim)

    return a.reshape(dim), newshape


# =========================================================
def getCRF_justcol(img, Lc, theta, n_iter, label_lines, compat_col=40, scale=5, prob=0.5):

      if np.ndim(img)==2:
         img = np.dstack((img, img, img))
	     
      H = img.shape[0]
      W = img.shape[1]

      d = dcrf.DenseCRF2D(H, W, len(label_lines)+1)
      U = unary_from_labels(Lc.astype('int'), len(label_lines)+1, gt_prob= prob)

      d.setUnaryEnergy(U)

      del U

      # sdims = The scaling factors per dimension.
      # schan = The scaling factors per channel in the image.
      # This creates the color-dependent features and then add them to the CRF
      feats = create_pairwise_bilateral(sdims=(theta, theta), schan=(scale, scale, scale), #11,11,11
                                  img=img, chdim=2)

      del img

      d.addPairwiseEnergy(feats, compat=compat_col,
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)

      del feats
      Q = d.inference(n_iter)

      preds = np.array(Q, dtype=np.float32).reshape(
        (len(label_lines)+1, H, W)).transpose(1, 2, 0)
      preds = np.expand_dims(preds, 0)
      preds = np.squeeze(preds)

      return np.argmax(Q, axis=0).reshape((H, W)), preds #, p, R, np.abs(d.klDivergence(Q)/ (H*W))



#==============================================================================

# mouse callback function
def anno_draw(event,former_x,former_y,flags,param):
    global current_former_x,current_former_y,drawing, mode, lw

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        current_former_x,current_former_y=former_x,former_y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),lw) #10, 5)
                current_former_x = former_x
                current_former_y = former_y
                #print former_x,former_y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(im,(current_former_x,current_former_y),(former_x,former_y),(0,0,255),lw) #5)
            current_former_x = former_x
            current_former_y = former_y
    return former_x,former_y


#==============================================================================
# #==============================================================================
# def get_img(image_path, fct):
   # img = cv2.imread(image_path)
   # if fct<1:
      # img = cv2.resize(img, (0,0), fx=fct, fy=fct) 

   # img[img==0] = 1
   # img[img==255] = 254

   # nxo, nyo, nz = np.shape(img)
   # # pad image so it is divisible by N windows with no remainder
   # return np.pad(img, [(0,win-np.mod(nxo,win)), (0,win-np.mod(nyo,win)), (0,0)], mode='constant')

   
#==============================================================================
def resize_img(img, fct):
   #if fct<1:
   img = cv2.resize(img, (0,0), fx=fct, fy=fct) 

   if np.ndim(img)==2:
      img = np.dstack((img, img, img))
   
   img[img==0] = 1
   img[img==255] = 254
   # pad image so it is divisible by N windows with no remainder

   if np.ndim(img)==3:
      nxo, nyo, nz = np.shape(img)
      pimg = np.pad(img, [(0,win-np.mod(nxo,win)), (0,win-np.mod(nyo,win)), (0,0)], mode='constant')
   else:
      nxo, nyo = np.shape(img)
      pimg = np.pad(img, [(0,win-np.mod(nxo,win)), (0,win-np.mod(nyo,win))], mode='constant')
   return pimg

   
#==============================================================
if __name__ == '__main__':

   argv = sys.argv[1:]
   try:
      opts, args = getopt.getopt(argv,"h:w:s:")
   except getopt.GetoptError:
      print('python label_1geotiff_crf.py -w windowsize -s size')
      sys.exit(2)

   for opt, arg in opts:
      if opt == '-h':
         print('Example usage: python label_1geotiff_crf.py -w 400 -s 0.125')
         sys.exit()
      elif opt in ("-w"):
         win = arg
      elif opt in ("-s"):
         fct = arg

		 
   #===============================================
   # Run main application
   Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing   
   image_path = askopenfilename(filetypes=[("pick an image file","*.tif *.tiff *.TIF *.TIFF")], multiple=False)  

   labels_path = askopenfilename(filetypes=[("pick a labels file","*.txt")], multiple=False)  
   colors_path = askopenfilename(filetypes=[("pick a label colors file","*.txt")], multiple=False)  
   
   #hostname = socket.gethostname()

   name, ext = os.path.splitext(image_path)
   name = name.split(os.sep)[-1]   
   
   # start timer
   if os.name=='posix': # true if linux/mac or cygwin on windows
      start = time.time()
   else: # windows
      start = time.clock()
      
   win = int(win) ##1000
   fct = float(fct) ##1000

   lw = 5 #initial brush thickness
   print("initial brush width = "+str(lw))
   print("change using the +/- keys")
   print("cycle classes with [ESC]")
   
   theta=60
   compat_col=100 
   scale=1
   n_iter=30

   with open(labels_path) as f: #'labels.txt') as f:
      labels = f.readlines()
   labels = [x.strip() for x in labels] 

   with open(colors_path) as f: #'labels.txt') as f:
      cols = f.readlines()
   cmap1 = [x.strip() for x in cols] 
 
   classes = dict(zip(labels, cmap1))

   cmap = colors.ListedColormap(cmap1)

   #===============================================

   drawing=False # true if mouse is pressed
   mode=True # if True, draw rectangle. Press 'm' to toggle to curve

   img, bs = read_geotiff(image_path)
   img = resize_img(img, fct)
   #img = get_img(image_path, fct)
   
   if np.ndim(img)==3:   
      nx, ny, nz = np.shape(img)
      Z,ind = sliding_window(img, (win, win,3), (win, win,3))	  
   else:
      nx, ny = np.shape(img)   
      Z,ind = sliding_window(img, (win, win), (win, win))

   gridy, gridx = np.meshgrid(np.arange(ny), np.arange(nx))

   Zx,_ = sliding_window(gridx, (win, win), (win, win))
   Zy,_ = sliding_window(gridy, (win, win), (win, win))

   out = np.zeros((nx,ny))    
			
   for ck in range(len(Z)):
      #img = get_img(image_path, fct)
      img, bs = read_geotiff(image_path)
      img = resize_img(img, fct)	  
	  
      cv2.rectangle(img, (np.min(Zy[ck]), np.min(Zx[ck])), (np.max(Zy[ck]), np.max(Zx[ck])), (255,0,0), 2)	  
      cv2.namedWindow('whole image')			
      cv2.imshow('whole image',img)
	  
      im = Z[ck].copy()
      cim = Z[ck].copy()	  
      counter=1
      if np.std(im)>0:
         for label in labels:
            imcopy = im.copy()		    
            conf = 0
            #=============================
            cv2.namedWindow(label) #+' ('+str(ck+1)+'/'+str(len(Z))+')')#, cv2.WND_PROP_FULLSCREEN) 
            cv2.moveWindow(label, 0,0)  # Move it to (0,0)
            #cv2.setWindowProperty(label, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            cv2.setMouseCallback(label,anno_draw)
            while(1):
               cv2.imshow(label,im)
               k=cv2.waitKey(1)&0xFF
               if k==122: #'Z'
                  im = imcopy.copy()
                  imcopy = im.copy()				  
               if k==27:
                  try:		   
                     im[im[:,:,2]==255] = counter
                  except:
                     im[im==255] = counter
				  
                  counter += 1
                  break
               #plus = 43
               if k==43:
                  lw += 1
                  print("brush width = "+str(lw))
               #minus = 45
               if k==45:
                  lw -= 1
                  if lw<1:
                     lw=1
                  print("brush width = "+str(lw))			  
            cv2.destroyWindow(label) #destroyAllWindows()

         try:
            Lc = im[:,:,2]
         except:
            Lc = im[:,:]
			
         Lc[Lc>=counter] = 0

         out[Zx[ck],Zy[ck]] = Lc
		 		 
      else:
         try:	  
            Lc = np.zeros(np.shape(im[:,:,2]))
         except:		 
            Lc = np.zeros(np.shape(im))
         
         out[Zx[ck],Zy[ck]] = Lc
		 
      cv2.destroyWindow('whole image')

			
   #==========================
   #img = get_img(image_path, fct) 

   img, bs = read_geotiff(image_path)
   #img = resize_img(img, fct)   
   
   try:
      nxo, nyo, nz = np.shape(img)
   except:
      nxo, nyo = np.shape(img)
   

   Lc = out[:nxo,:nyo] 

   im = img[:nxo, :nyo]
   
   #b,g,r = cv2.split(im)       # get b,g,r
   #rgb_img = cv2.merge([r,g,b])     # switch it to rgb
   
   
   print('Generating dense scene from sparse labels ....')
   res,p = getCRF_justcol(im, Lc.astype('int'), theta, n_iter, classes, compat_col, scale)

   resr = np.round(imresize(res, 1/fct, interp='nearest')/255 * np.max(res)).astype('int')
   Lcr = np.round(imresize(Lc, 1/fct, interp='nearest')/255 * np.max(Lc)).astype('int')

   #im = cv2.imread(image_path)   
   #b,g,r = cv2.split(im)       # get b,g,r
   #rgb_img = cv2.merge([r,g,b])     # switch it to rgb
   
   try:
      nxo, nyo, nz = np.shape(img)
   except:
      nxo, nyo = np.shape(img)   
   
   Lcr = Lcr[:nxo,:nyo] 
   resr = resr[:nxo,:nyo]    
   
   resr = median(resr, disk(5))
   
   savemat(image_path.split('.')[0]+'_mres.mat', {'sparse': Lcr.astype('int'), 'class': resr.astype('int'), 'preds': p.astype('float16'), 'labels': labels}, do_compression = True) 

   Lcorig = Lcr.copy().astype('float')
   Lcorig[Lcorig<1] = np.nan
   

   #=============================================   
   #=============================================
   print('Generating plot ....')
   fig = plt.figure()
   fig.subplots_adjust(wspace=0.4)
   ax1 = fig.add_subplot(131)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   _ = ax1.imshow(im)
   #plt.title('a) Input', loc='left', fontsize=6)

   #=============================   
   ax1 = fig.add_subplot(132)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   _ = ax1.imshow(im)
   #plt.title('b) Unary potentials', loc='left', fontsize=6)
   im2 = ax1.imshow(Lcorig-1, cmap=cmap, alpha=0.5, vmin=0, vmax=len(labels))
   divider = make_axes_locatable(ax1)
   cax = divider.append_axes("right", size="5%")
   cb=plt.colorbar(im2, cax=cax)
   cb.set_ticks(0.5+np.arange(len(labels)+1))
   cb.ax.set_yticklabels(labels)
   cb.ax.tick_params(labelsize=4)

   #=============================
   ax1 = fig.add_subplot(133)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)   

   _ = ax1.imshow(im)
   #plt.title('c) CRF prediction', loc='left', fontsize=6)
   im2 = ax1.imshow(resr, cmap=cmap, alpha=0.5, vmin=0, vmax=len(labels))
   divider = make_axes_locatable(ax1)
   cax = divider.append_axes("right", size="5%")
   cb=plt.colorbar(im2, cax=cax)
   cb.set_ticks(0.5+np.arange(len(labels)+1))
   cb.ax.set_yticklabels(labels)
   cb.ax.tick_params(labelsize=4)
   plt.savefig(name+'_mres.png', dpi=600, bbox_inches='tight')
   del fig; plt.close()
   
   #=============================================   
   #=============================================
   if os.name=='posix': # true if linux/mac
      elapsed = (time.time() - start)
   else: # windows
      elapsed = (time.clock() - start)
   print("Processing took "+ str(elapsed/60) + "minutes")
   
   # write report
   file = open(name+'_report_'+socket.gethostname()+'.txt','w') 
   file.write('Image: '+image_path+'\n')
   counter = 0
   for label in labels:
      file.write(label+': '+str(np.sum(resr==counter)/(nxo*nyo))+'\n')
      counter += 1	  
   file.write('Processing time (mins): '+str(elapsed/60)+'\n')   
   file.close()   
  
   #==========================================================
   try:
      epsg = int(bs[0]['crs']['init'].split('epsg:')[-1])
      proj = osr.SpatialReference()
      proj.ImportFromEPSG(epsg) 	  
   except:
      d = gdal.Open(image_path)
      proj = osr.SpatialReference(wkt=d.GetProjection())  

   datout = np.squeeze(np.ma.filled(1+resr.copy()))

   datout[np.isnan(datout)] = 0
   datout = datout.astype('int')
   driver = gdal.GetDriverByName('GTiff')
   cols,rows = np.shape(datout)    

   outFile = os.path.normpath(name+'mres_geo.tif')
   ds = driver.Create( outFile, rows, cols, 1, gdal.GDT_UInt16, [ 'COMPRESS=LZW' ] )        
   if proj is not None:  
     ds.SetProjection(proj.ExportToWkt()) 

   #xmin, ymin, xmax, ymax = [bs[0]['lonmin'], bs[0]['latmin'], bs[0]['lonmax'], bs[0]['latmax']]
   xmin, ymin, xmax, ymax = [bs[0]['xmin'], bs[0]['ymin'], bs[0]['xmax'], bs[0]['ymax']]

   xres = (xmax - xmin) / float(rows)
   yres = (ymax - ymin) / float(cols)

   geotransform = (xmin, xres, 0, ymax, 0, -yres)

   ds.SetGeoTransform(geotransform)
   ss_band = ds.GetRasterBand(1)
   ss_band.WriteArray(datout) 
   ss_band.SetNoDataValue(0) #-99)
   ss_band.FlushCache()
   ss_band.ComputeStatistics(False)
   del ds      
   
   
   
