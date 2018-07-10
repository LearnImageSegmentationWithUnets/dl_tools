from __future__ import division
import rasterio
from scipy.ndimage import zoom
import numpy as np
import pyproj
from imageio import imwrite

## conda install rasterio pyproj

###C:\Users\ddb265\github_clones\ms_backscatter_comp\20160331_Bedford\Bedford16_StaticSurface.tif

##-------------------------------------------------------------
def read_geotiff(input, gridres):
   """
   This function reads image in GeoTIFF format. Optionally resizes
   """
   ## input = list of strings of filenames
   ##         gridres = grd resolution in m

   print('Reading GeoTIFF data ...')
   if type(input) is not list:
      input = [input]

   ## read all arrays
   bs = []
   for layer in input:
      with rasterio.open(layer) as src:
         layer = src.read()[0,:,:]
      w, h = (src.width, src.height)
      xmin, ymin, xmax, ymax = src.bounds
      crs = src.get_crs()
      del src
      bs.append({'bs':layer, 'w':w, 'h':h, 'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax, 'crs':crs})

   #get pyproj transformation object
   trans = pyproj.Proj(init=bs[0]['crs']['init'])

   ## resize arrays so common grid
   ##get bounds
   xmax = max([x['xmax'] for x in bs])
   xmin = min([x['xmin'] for x in bs])
   ymax = max([x['ymax'] for x in bs])
   ymin = min([x['ymin'] for x in bs])
   ## make common grid
   yp, xp = np.meshgrid(np.arange(xmin, xmax, gridres), np.arange(ymin, ymax, gridres))

   ## get extents in lat/lon
   lonmin, latmin = trans(xmin, ymin, inverse=True)
   lonmax, latmax = trans(xmax, ymax, inverse=True)

   nx, ny = np.shape(yp)
   for k in range(len(bs)):
      bs[k]['bs'] = zoom(bs[k]['bs'], (nx/bs[k]['h'], ny/bs[k]['w']))
      bs[k]['h'] = nx
      bs[k]['w'] = ny
      bs[k]['xmin'] = xmin
      bs[k]['xmax'] = xmax
      bs[k]['ymin'] = ymin
      bs[k]['ymax'] = ymax
      bs[k]['latmin'] = latmin
      bs[k]['latmax'] = latmax
      bs[k]['lonmin'] = lonmin
      bs[k]['lonmax'] = lonmax
      bs[k]['trans'] = trans
      bs[k]['gridres'] = gridres

   img = np.dstack([x['bs'] for x in bs]).astype('uint8')

   return np.squeeze(img), bs
