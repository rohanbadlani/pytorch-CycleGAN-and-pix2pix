#!/bin/bash
# Usage: bash zip_test.sh <model_name>
# Example: bash zip_test.sh perceptual_pix2pix

RESULT_DIR=~/turning-dreams-to-reality/pytorch-CycleGAN-and-pix2pix/results/$1/test_latest
mkdir -p $RESULT_DIR/$1_test_images
cp $RESULT_DIR/images/*fake* $RESULT_DIR/$1_test_images
zip -r $RESULT_DIR/$1_test_images.zip $RESULT_DIR/$1_test_images 

echo "Zip file location:"
echo $RESULT_DIR/$1_test_images.zip
