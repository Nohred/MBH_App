#!/bin/bash
# run_experiments.sh

echo "Entrenando modelo UNet por 50 epocas..."
python train.py --model unet --epochs 50 --lr 2e-4



