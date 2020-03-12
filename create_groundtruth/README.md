# Labeling

This is a tool for image classification written by Dr Daniel Buscombe Northern
Arizona University https://github.com/dbuscombe-usgs/dl_tools

It was edited for use in a thesis at NPS and posted here in case the edits can
help others. The labeling was broken out of the rest of the pipeline for three
reasons:
* The labels created can be used for a wide variety of ML applications
* It's much easier to do this on a computer that has a touchscreen and a stylus
* It cannot be done on the HPC through jupyter notebook while the rest can

## Instructions for use:

### Clone
```
git clone https://gitlab.nps.edu/bbwells/labeling
```

### UNTAR or UNZIP
```
code to untar
```

### cd into the folder
```
cd labeling
```

### create environment
Note: This environment does not contain tensoflow.
```
conda import label.yml
```

### Activate environment
```
conda env create -f label.yml
```

### Add pictures
# TODO: note where to put pictures
Add the pictures you would like to classify into the /data/images/ folder

### Change colors and labels
It's set in the params file within modules, you may have to play with the
colors a bit. This tool should not be used with images that have more than ~8
labels. It can be done, but you should consider doing it multiple projects
to get the labeled images you need.

#TODO: make a tool to play with the colors without needsing to relabel an
# an entire image

### Change the import line to your project name
Within the imports section of Labeler.py there is a line that reads like this:
```
from modules.params_sat import params
```
It is that way, so that you can set different parameter files for every
project. Then, Labeler will switch which folder, labels, colors, etc. Change
the 'params_sat' to the name of the parameters file you would like to use.

### Run
If you are running this through command line, you will need to cd to the
directory you will be working in:
```
cd /Documents/Labeler
```
Then run it with:
```
python Labeler.py
```
It will select the next image the next time you run it.

### Draw on the image
The title of the window is the label that will be associated with the pixels
you draw on. After you are done with label press escape. You can increae and
decrease the brush width with + / -. You can also undo a mistake with z.

### Redo an image
This has a slight learning curve to it, so you may wish to
redo an image. There is a list of done image names in the
images folder specified by your params.py file. Just delete
a name to redo it.

### Do the CRF later or somewhere else
After you get the hang of it, you may wish to just draw for
a few hours without waiting for the crf to complete each time.
You can set the "do_dense_later" variable in params to True,
and it will leave the npy files to do later. Just run CRFLater.py
If you want to do this on another machine, transfer these files:
  images of interest
  npy files of interest
  Crf.py
  params.py
  CRFLater.py
  PlotAndSave.py
all with the same folder structure set in params.py

## Problems or suggestions?
If you would like to add the ability to use 5 band images, or geotiffs email me
or create a pull request.

While this should work with any size image, it has only been tested using
pictures of ice with dimensions of 2008x3008. If you are having issues with your
pictures please email me or create a pull request. I am intimately familiar with
this code now and should be able to update it quickly.
