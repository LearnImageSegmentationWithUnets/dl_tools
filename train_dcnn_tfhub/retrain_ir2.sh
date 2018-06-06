## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

##USAGE:
##bash retrain_ir2.sh "C:\Users\ddb265\Desktop\imrecog_data\NWPU-RESISC45\train" 1000 0.01

DIREC=$1
NUM_STEPS=$2
LEARNRATE=$3

start=`date +%s`

/c/Users/ddb265/AppData/Local/Continuum/anaconda3/envs/tfpy35/python retrain.py --image_dir $DIREC \
    --tfhub_module https://tfhub.dev/google/imagenet/inception_resnet_v2/classification/1 \
    --how_many_training_steps $NUM_STEPS --learning_rate $LEARNRATE --output_labels labels.txt --output_graph irv2_"$NUM_STEPS"_"$LEARNRATE".pb

rm -rf /c/tmp/bottleneck
rm -rf /c/tmp/checkpoint
rm -rf /c/tmp/retrain_logs
rm -rf /c/tmp/_retrain*

end=`date +%s`

runtime=$((end-start))

echo "Execution time:"
echo $runtime" seconds"
