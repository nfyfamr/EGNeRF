# #!/bin/bash

export ROOT_DIR=/home/ec2-user/data/ShapeNet_SRN/srn_cars

EXP=srn_cars/hashgrid11_batch4096_fgen1024_embed256_lr34_34_bias_feat
python train.py \
    --root_dir $ROOT_DIR/cars_train_2457 \
    --exp_name $EXP/train_2457 \
    --dataset_name srn --num_epochs 20 --batch_size 4096 --max_samples 128 \
    --lr 1e-3 --lr_decay 1e-1 --emb_lr 1e-3 --emb_lr_decay 1e-1 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 256 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2


python optimize.py \
    --root_dir $ROOT_DIR/cars_test_60 \
    --exp_name $EXP/34/test_60 \
    --dataset_name srn --num_epochs 200 --batch_size 4096 --max_samples 128 \
    --emb_lr 1e-3 --emb_lr_decay 1e-1 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 256 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/$EXP/train_2457/epoch=19.ckpt \
    --view_idxs 5 --split test_opt --embed_bias_feat
