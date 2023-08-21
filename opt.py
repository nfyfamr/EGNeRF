import argparse

def get_opts():
    parser = argparse.ArgumentParser()

    # dataset parameters
    parser.add_argument('--root_dir', action='append', type=str, required=True,
                        help='root directory of dataset')
    parser.add_argument('--dataset_name', type=str, default='nsvf',
                        choices=['nerf', 'nsvf', 'colmap', 'nerfpp', 'rtmv', 'srn'],
                        help='which dataset to train/test')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'trainval', 'trainvaltest'],
                        help='use which split to train')
    parser.add_argument('--downsample', type=float, default=1.0,
                        help='downsample factor (<=1.0) for the images')

    # model parameters
    parser.add_argument('--scale', type=float, default=0.5,
                        help='scene scale (whole scene must lie in [-scale, scale]^3')
    parser.add_argument('--use_exposure', action='store_true', default=False,
                        help='whether to train in HDR-NeRF setting')

    # loss parameters
    parser.add_argument('--distortion_loss_w', type=float, default=0,
                        help='''weight of distortion loss (see losses.py),
                        0 to disable (default), to enable,
                        a good value is 1e-3 for real scene and 1e-2 for synthetic scene
                        ''')

    # training options
    parser.add_argument('--batch_size', type=int, default=8192,
                        help='number of rays in a batch')
    parser.add_argument('--ray_sampling_strategy', type=str, default='all_images',
                        choices=['all_images', 'same_image'],
                        help='''
                        all_images: uniformly from all pixels of ALL images
                        same_image: uniformly from all pixels of a SAME image
                        ''')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of training epochs')
    parser.add_argument('--num_gpus', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1e-2,
                        help='learning rate decay; final learning rate: lr*lr_decay')
    # experimental training options
    parser.add_argument('--optimize_ext', action='store_true', default=False,
                        help='whether to optimize extrinsics')
    parser.add_argument('--random_bg', action='store_true', default=False,
                        help='''whether to train with random bg color (real scene only)
                        to avoid objects with black color to be predicted as transparent
                        ''')

    # validation options
    parser.add_argument('--eval_lpips', action='store_true', default=False,
                        help='evaluate lpips metric (consumes more VRAM)')
    parser.add_argument('--val_only', action='store_true', default=False,
                        help='run only validation (need to provide ckpt_path)')
    parser.add_argument('--no_save_test', action='store_true', default=False,
                        help='whether to save test image and video')

    # misc
    parser.add_argument('--exp_name', type=str, default='exp',
                        help='experiment name')
    parser.add_argument('--ckpt_path', type=str, default=None,
                        help='pretrained checkpoint to load (including optimizers, etc)')
    parser.add_argument('--weight_path', type=str, default=None,
                        help='pretrained checkpoint to load (excluding optimizers, etc)')

    # network config
    parser.add_argument('--grid', type=str, default='Hash',
                        choices=['Hash', 'Window', "MixedFeature"],
                        help='Encoding scheme Hash or MixedFeature')
    parser.add_argument('--L', type=int, default=16,
                        help='Encoding hyper parameter L')
    parser.add_argument('--F', type=int, default=2,
                        help='Encoding hyper parameter F')
    parser.add_argument('--T', type=int, default=19,
                        help='Encoding hyper parameter T')
    parser.add_argument('--N_min', type=int, default=16,
                        help='Encoding hyper parameter N_min')
    parser.add_argument('--N_max', type=int, default=2048,
                        help='Encoding hyper parameter N_max')
    parser.add_argument('--N_tables', type=int, default=1,
                        help='Number of hash tables')
    parser.add_argument('--max_samples', type=int, default=1024,
                        help='Number of sample points per ray')
    
    parser.add_argument('--rgb_channels', type=int, default=64,
                        help='rgb network channels')
    parser.add_argument('--rgb_layers', type=int, default=2,
                        help='rgb network layers')

    parser.add_argument('--seed', type=int, default=1337,
                        help='random seed')
    
    # multi scene training config
    parser.add_argument('--embed_size', type=int, default=32,
                        help='size of scene embedding vector')
    parser.add_argument('--embed_mode', type=str, default='sum',
                        choices=["sum", "concat"],
                        help='scene embedding blending mode')
    
    parser.add_argument('--hyper', action='store_true', default=False,
                        help='run feature grid geneartion code')
    parser.add_argument('--fgen_channels', type=int, default=512,
                        help='feature generator network channels')
    parser.add_argument('--fgen_layers', type=int, default=2,
                        help='feature generator network layers')

    return parser.parse_args()
