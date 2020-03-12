# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 21:40:19 2019

@author: Brodie

A list of changable parameters for use with labeling. A description of each
is before it.
"""

# =============================================================================
"""
All these paths and folders should exist from the git, if not, please create
 them. Or, change them here.
"""
images_path = './data_jd/im_to_do/'
# Images to do, it will cycle through these
done_im_pat = './data_jd/im_done/'
# After images have a npy file made, they will be moved here move them
#   back to images_path if you would like to redo them
result_path = './data_jd/label/'
# Where you want label image to go
compar_path = './data_jd/comparison/'
# Where you want to comparison plots
matlab_path = './data_jd/matFile/'
# Where you want to keep the matlab files
report_path = './data_jd/report/'
# Where you want the reports
unsplit_pat = './data_jd/im_to_do/unsplit/'
# If the image is too big, it will be split up, where would you like to
# save the original?
den_lat_pat = './data_jd/dense_later/'
# If you choose to do the crf later where would you like the
# npy files saved?
image_types = [images_path + '*.TIF', images_path + '*.tif',
               images_path + '*.tiff',
               images_path + '*.JPG', images_path + '*.jpg',
               images_path + '*.jpeg', images_path + '*.JPEG',
               images_path + '*.png', images_path + '*.PNG'
               ]
# List of image types to look for

# =============================================================================
"""
Set these for the drawing GUI
"""
max_x_steps = 6
max_y_steps = 6
ref_im_scale = .7
# scale down the whole image for viewing use 1 if your
# image fits on your screen
# TODO: I don't think this is being used properly
lw = 5
# Initial line width
im_order = 'RGB'
# Some tif images are in BGR others are RGB
#   -options are: 'BGR', 'RGB'

# =============================================================================
"""
To change how the results are output.
"""
a_label = 'a) Input'
# What to label the left image
b_label = 'b) CRF prediction'
# What to label the middle image, or right image if doing 2
plot_c = False
# Do you want to plot your markings?
c_label = 'b) CRF prediction'
# What to label the right image
font_size = 6
# Font size for results images
alpha_percent = 0.4
# How see through the results should be (0-1)
classes = {'Sand': '#FF0000',
           'Algae': '#329ba7',
           'Road': '#704515',
           'Shrubery': '#c8e67a',
           'Water': '#d78b05',
           'BackGrnd': '#FFFFFF'}
# Set classes and colors here
auto_class = "No"
# Do you have a class that will be ignored by the machine learning? Like,
#  Shadow, or BackGround? In some cases all pixels in a small frame are 0,
#  and it will automatically set those to the throw away class name. If you
#  do not have one, set this to "No"

# =============================================================================
"""
These parameters are for the CRF. You will likely not want to change these.
They work well with any images I have tried for this purpose. Although some
can be learned and a good project would be writing a machine learning node
to learn what works best.
"""
theta = 60
# "nearness" tolerance
n_iter = 100
# "intensity" tolerance
compat_col = 40
scale = 5
prob = 0.5
do_dense_later = True
# If you get into a groove and just want to label for a while, you
#   can save off the labels to do the dense CRF portion later

# =============================================================================
"""
If you are doing a gray-scale image, sometimes it is best
to look at it with a different colormap. Other optins found at:
    https://matplotlib.org/tutorials/colors/colormaps.html
"""
c_map = 'gray'

# =============================================================================
"""
This just puts it all in one place.  No need to edit this unless you add or
  subtract a whole parameter above.
"""
params = {'images_path': images_path,
          'done_im_pat': done_im_pat,
          'result_path': result_path,
          'compar_path': compar_path,
          'matlab_path': matlab_path,
          'report_path': report_path,
          'unsplit_pat': unsplit_pat,
          'den_lat_pat': den_lat_pat,
          'image_types': image_types,
          'max_x_steps': max_x_steps,
          'max_y_steps': max_y_steps,
          'ref_im_scale': ref_im_scale,
          'lw': lw,
          'im_order': im_order,
          'a_label': a_label,
          'b_label': b_label,
          'plot_c': plot_c,
          'c_label': c_label,
          'font_size': font_size,
          'alpha_percent': alpha_percent,
          'classes': classes,
          'auto_class': auto_class,
          'theta': theta,
          'n_iter': n_iter,
          'compat_col': compat_col,
          'scale': scale,
          'prob': prob,
          'do_dense_later': do_dense_later,
          'c_map': c_map}
