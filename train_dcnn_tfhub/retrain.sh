
# rm -rf /c/tmp/bottleneck
# rm -rf /c/tmp/checkpoint
# rm -rf /c/tmp/retrain_logs
# rm -rf /c/tmp/_retrain*

/c/Users/ddb265/AppData/Local/Continuum/anaconda3/envs/tfpy35/python retrain.py --image_dir aclass_OB96 \
    --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_96/classification/1 \
    --how_many_training_steps 4000 --output_labels labels.txt --output_graph mobilenetv2_96_graph_b.pb

rm -rf /c/tmp/bottleneck
rm -rf /c/tmp/checkpoint
rm -rf /c/tmp/retrain_logs
rm -rf /c/tmp/_retrain*
