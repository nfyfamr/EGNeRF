# #!/bin/bash

export ROOT_DIR=/home/ec2-user/data/ShapeNet_SRN/srn_cars

# python train.py \
#     --root_dir $ROOT_DIR/cars_train \
#     --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
#     --dataset_name srn --num_epochs 20 --batch_size 16384 --max_samples 128 \
#     --lr 2e-3 --lr_decay 1e-2 --eval_lpips \
#     --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
#     --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
#     --hyper --fgen_channels 1024 --fgen_layers 2


python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-1 --lr_decay 1e-1 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt

python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-1 --lr_decay 1e-2 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt


python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-1 --lr_decay 1e-3 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt

python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-1 --lr_decay 1e-4 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt

python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-1 --lr_decay 1e-5 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt



python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-2 --lr_decay 1e-1 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt

python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-2 --lr_decay 1e-2 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt


python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-2 --lr_decay 1e-3 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt

python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-2 --lr_decay 1e-4 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt




python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-3 --lr_decay 1e-1 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt

python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-3 --lr_decay 1e-2 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt


python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-3 --lr_decay 1e-3 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt





python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-4 --lr_decay 1e-1 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt

python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-4 --lr_decay 1e-2 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt



python optimize.py \
    --root_dir $ROOT_DIR/cars_test \
    --exp_name Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch \
    --dataset_name srn --num_epochs 200 --batch_size 16384 --max_samples 128 \
    --lr 2e-5 --lr_decay 1e-1 --eval_lpips \
    --L 16 --F 2 --T 11 --N_min 16 --grid Hash \
    --rgb_channels 64 --rgb_layers 2 --embed_size 64 --embed_mode concat \
    --hyper --fgen_channels 1024 --fgen_layers 2 \
    --ckpt_path ckpts/srn/Test/srn_cars/hashgrid_T11_levels_16_F_2_hyper_2458_samples_128_2ly_1024ch/epoch=19.ckpt \
    --view_idxs 5 --split test_opt
