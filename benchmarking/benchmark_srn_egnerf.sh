# #!/bin/bash

export ROOT_DIR=/home/ubuntu/data/ShapeNet_SRN/srn_cars

python train.py \
    --root_dir $ROOT_DIR/cars_train_100 \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2000_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 20 --batch_size 16384 --max_samples 128 \
    --lr 2e-3 --lr_decay 1e-2 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2
