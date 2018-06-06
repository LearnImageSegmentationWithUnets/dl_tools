## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

##USAGE:
##bash retrain_m2.sh "C:\Users\ddb265\Desktop\imrecog_data\NWPU-RESISC45\train" 224 1000 0.01

DIREC=$1
TILESIZE=$2
NUM_STEPS=$3
LEARNRATE=$4

start=`date +%s`

/c/Users/ddb265/AppData/Local/Continuum/anaconda3/envs/tfpy35/python retrain.py --image_dir $DIREC \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_"$TILESIZE"/classification/1 \
    --how_many_training_steps $NUM_STEPS --learning_rate $LEARNRATE --output_labels labels.txt --output_graph mobilenetv2_"$TILESIZE"_"$NUM_STEPS"_"$LEARNRATE".pb

rm -rf /c/tmp/bottleneck
rm -rf /c/tmp/checkpoint
rm -rf /c/tmp/retrain_logs
rm -rf /c/tmp/_retrain*

end=`date +%s`

runtime=$((end-start))

echo "Execution time:"
echo $runtime" seconds"
