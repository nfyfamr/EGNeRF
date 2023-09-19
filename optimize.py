import torch
from torch import nn
from opt import get_opts
import os
import glob
import imageio
import numpy as np
import cv2
from einops import rearrange

# data
from torch.utils.data import DataLoader
from datasets import dataset_dict
from datasets.ray_utils import axisangle_to_R, get_rays

# models
from kornia.utils.grid import create_meshgrid3d
from models.networks import NGP
from models.rendering import render, MAX_SAMPLES
from models import rendering

# optimizer, losses
from apex.optimizers import FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from losses import NeRFLoss

# metrics
from torchmetrics import (
    PeakSignalNoiseRatio, 
    StructuralSimilarityIndexMeasure
)
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

# pytorch-lightning
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import all_gather_ddp_if_available

from utils import slim_ckpt, load_ckpt

import warnings; warnings.filterwarnings("ignore")


def depth2img(depth):
    depth = (depth-depth.min())/(depth.max()-depth.min())
    depth_img = cv2.applyColorMap((depth*255).astype(np.uint8),
                                  cv2.COLORMAP_TURBO)

    return depth_img


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        self.warmup_steps = 256
        self.update_interval = 16

        if self.hparams.dataset_name == 'srn' and self.hparams.hyper:
            if len(self.hparams.root_dir) != 1:
                raise Exception("srn dataset should have one root_dir")
            # Expected path: data/ShapeNet_SRN/srn_cars/cars_train
            self.root_dirs = [os.path.join(self.hparams.root_dir[0], d) for d in os.listdir(self.hparams.root_dir[0])]
        else:
            self.root_dirs = self.hparams.root_dir
        
        self.num_scene = len(self.root_dirs)
        hparams.num_scene = self.num_scene
        if hparams.view_idxs is None: hparams.view_idxs = [0]
        hparams.num_views = self.num_views = len(hparams.view_idxs)

        rendering.MAX_SAMPLES = self.hparams.max_samples

        self.loss = NeRFLoss(lambda_distortion=self.hparams.distortion_loss_w,
                            lambda_bce=self.hparams.bce_loss_w)
        self.train_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_psnr = PeakSignalNoiseRatio(data_range=1)
        self.val_ssim = StructuralSimilarityIndexMeasure(data_range=1)
        if self.hparams.eval_lpips:
            self.val_lpips = LearnedPerceptualImagePatchSimilarity('vgg')
            for p in self.val_lpips.net.parameters():
                p.requires_grad = False

        rgb_act = 'None' if self.hparams.use_exposure else 'Sigmoid'
        self.model = NGP(scale=self.hparams.scale, 
                            hparams=hparams,
                            rgb_act=rgb_act)
        print(f"T: {hparams.T}")
        print(f"{self.model.density_net.native_tcnn_module.n_params() - 3072} 3072 {self.model.rgb_net.native_tcnn_module.n_params()}")

        G = self.model.grid_size
        self.model.register_buffer('density_grid',
            torch.zeros(self.model.cascades, G**3))
        self.model.register_buffer('grid_coords',
            create_meshgrid3d(G, G, G, False, dtype=torch.int32).reshape(-1, 3))

    def forward(self, batch, split):
        scn_idx = torch.tensor(batch['scn_idx'], device=batch['rgb'].device)
        if split=='train':
            poses = self.poses[scn_idx][batch['img_idxs']]
            directions = self.directions[scn_idx, batch['pix_idxs']]
        else:
            poses = batch['pose']
            directions = self.directions[scn_idx]

        if self.hparams.optimize_ext:
            dR = axisangle_to_R(self.dR[batch['img_idxs']])
            poses[..., :3] = dR @ poses[..., :3]
            poses[..., 3] += self.dT[batch['img_idxs']]

        rays_o, rays_d = get_rays(directions, poses)

        kwargs = {'test_time': split!='train',
                  'random_bg': self.hparams.random_bg}
        if self.hparams.scale > 0.5:
            kwargs['exp_step_factor'] = 1/256
        if self.hparams.use_exposure:
            kwargs['exposure'] = batch['exposure']

        return render(self.model, rays_o, rays_d, scn_idx, **kwargs)

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]

        train_datasets = []
        for scn_idx, root_dir in enumerate(self.root_dirs):
            kwargs = {'root_dir': root_dir,
                    'downsample': self.hparams.downsample,
                    'scn_idx': scn_idx,
                    'view_idxs': self.hparams.view_idxs}
            ds = dataset(split=self.hparams.split, **kwargs)
            ds.batch_size = self.hparams.batch_size
            ds.ray_sampling_strategy = self.hparams.ray_sampling_strategy
            train_datasets.append(ds)
        self.train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        self.train_dataset.batch_size = self.hparams.batch_size
        self.train_dataset.ray_sampling_strategy = self.hparams.ray_sampling_strategy
        self.train_dataset.split = self.hparams.split
        self.train_dataset.img_wh = train_datasets[0].img_wh
        self.train_dataset.K = torch.stack([ds.K for ds in train_datasets])
        self.train_dataset.directions = torch.stack([ds.directions for ds in train_datasets])
        self.train_dataset.poses = torch.stack([ds.poses for ds in train_datasets])
        self.train_dataset.rays = torch.stack([ds.rays for ds in train_datasets])
        self.train_dataset.ds_sizes = [ds.poses.shape[0] for ds in train_datasets]

        test_datasets = []
        for scn_idx, root_dir in enumerate(self.root_dirs):
            kwargs = {'root_dir': root_dir,
                    'downsample': self.hparams.downsample,
                    'scn_idx': scn_idx,
                    'view_idxs': self.hparams.view_idxs}
            test_datasets.append(dataset(split='test', **kwargs))
        self.test_dataset = torch.utils.data.ConcatDataset(test_datasets)
        self.test_dataset.ds_sizes = [ds.poses.shape[0] for ds in test_datasets]

        self.dataloader_train = DataLoader(self.train_dataset,
                        num_workers=16,
                        persistent_workers=True,
                        batch_size=None,
                        pin_memory=True,
                        shuffle=True)

    def configure_optimizers(self):
        # define additional parameters
        self.register_buffer('directions', self.train_dataset.directions.to(self.device))
        self.register_buffer('poses', self.train_dataset.poses.to(self.device))

        if self.hparams.optimize_ext:
            N = len(self.train_dataset.poses)
            self.register_parameter('dR',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))
            self.register_parameter('dT',
                nn.Parameter(torch.zeros(N, 3, device=self.device)))

        load_ckpt(self.model, self.hparams.weight_path)

        emb_params = []
        for n, p in self.named_parameters():
            if n in ['model.scene_embed', 'model.scene_embed_bias_feat', 'model.scene_embed_bias_color']:
                emb_params += [p]

        opts = []
        self.emb_opt = FusedAdam(emb_params, self.hparams.emb_lr, eps=1e-15)
        opts += [self.emb_opt]
        if self.hparams.optimize_ext:
            opts += [FusedAdam([self.dR, self.dT], 1e-6)] # learning rate is hard-coded
        emb_sch = CosineAnnealingLR(self.emb_opt,
                                    self.hparams.num_epochs-1,
                                    self.hparams.emb_lr*self.hparams.emb_lr_decay)

        return opts, [emb_sch]

    def train_dataloader(self):
        return self.dataloader_train

    def val_dataloader(self):
        return DataLoader(self.test_dataset,
                          num_workers=8,
                          batch_size=None,
                          pin_memory=True)

    def on_train_start(self):
        # self.model.mark_invisible_cells(self.train_dataset.K.to(self.device),
        #                                 self.poses,
        #                                 self.train_dataset.img_wh)
        pass

    def training_step(self, batch, batch_nb, *args):
        # if self.global_step%self.update_interval == 0:
        #     self.model.update_density_grid(0.01*MAX_SAMPLES/3**0.5,
        #                                    warmup=self.global_step<self.warmup_steps,
        #                                    erode=self.hparams.dataset_name=='colmap')

        results = self(batch, split='train')
        loss_d = self.loss(results, batch)
        if self.hparams.use_exposure:
            zero_radiance = torch.zeros(1, 3, device=self.device)
            unit_exposure_rgb = self.model.log_radiance_to_rgb(zero_radiance,
                                    **{'exposure': torch.ones(1, 1, device=self.device)})
            loss_d['unit_exposure'] = \
                0.5*(unit_exposure_rgb-self.train_dataset.unit_exposure_rgb)**2
        loss = sum(lo.mean() for lo in loss_d.values())

        with torch.no_grad():
            self.train_psnr(results['rgb'], batch['rgb'])
        self.log('lr/emb_lr', self.emb_opt.param_groups[0]['lr'])
        self.log('train/loss', loss)
        # ray marching samples per ray (occupied space on the ray)
        self.log('train/rm_s', results['rm_samples']/len(batch['rgb']), True)
        # volume rendering samples per ray (stops marching when transmittance drops below 1e-4)
        self.log('train/vr_s', results['vr_samples']/len(batch['rgb']), True)
        self.log('train/psnr', self.train_psnr, True)

        return loss

    def on_validation_start(self):
        torch.cuda.empty_cache()
        if not self.hparams.no_save_test:
            self.val_dir = f'results/{self.hparams.dataset_name}/{self.hparams.exp_name}_opt'
            os.makedirs(self.val_dir, exist_ok=True)

    def validation_step(self, batch, batch_nb):
        rgb_gt = batch['rgb']
        results = self(batch, split='test')

        logs = {}
        # compute each metric per image
        self.val_psnr(results['rgb'], rgb_gt)
        logs['scn_idx'] = batch['scn_idx']
        logs['psnr'] = self.val_psnr.compute()
        self.val_psnr.reset()

        w, h = self.train_dataset.img_wh
        rgb_pred = rearrange(results['rgb'], '(h w) c -> 1 c h w', h=h)
        rgb_gt = rearrange(rgb_gt, '(h w) c -> 1 c h w', h=h)
        self.val_ssim(rgb_pred, rgb_gt)
        logs['ssim'] = self.val_ssim.compute()
        self.val_ssim.reset()

        if batch['img_idxs'] in self.hparams.view_idxs:
            self.logger.experiment.add_images(f'test/psnr/scene_{batch["scn_idx"]}', torch.concat([rgb_pred, rgb_gt]), 0)

        if self.hparams.eval_lpips:
            self.val_lpips(torch.clip(rgb_pred*2-1, -1, 1),
                           torch.clip(rgb_gt*2-1, -1, 1))
            logs['lpips'] = self.val_lpips.compute()
            self.val_lpips.reset()

        if not self.hparams.no_save_test: # save test image to disk
            idx = batch['img_idxs']
            rgb_pred = rearrange(results['rgb'].cpu().numpy(), '(h w) c -> h w c', h=h)
            rgb_pred = (rgb_pred*255).astype(np.uint8)
            depth = depth2img(rearrange(results['depth'].cpu().numpy(), '(h w) -> h w', h=h))
            imageio.imsave(os.path.join(self.val_dir, f'{batch["scn_idx"] if batch["scn_idx"] else 0}_{idx:04d}.png'), rgb_pred)
            imageio.imsave(os.path.join(self.val_dir, f'{batch["scn_idx"] if batch["scn_idx"] else 0}_{idx:04d}_d.png'), depth)

        return logs

    def validation_epoch_end(self, outputs):
        psnrs = torch.stack([x['psnr'] for x in outputs])
        mean_psnr = all_gather_ddp_if_available(psnrs).mean()
        self.log('test/psnr', mean_psnr, True)

        if self.num_scene != 1:
            # log per scene psnr
            outputs.sort(key=lambda x: x['scn_idx'])
            scene_psnrs = torch.stack([x['psnr'] for x in outputs])
            scene_psnrs = torch.split(scene_psnrs, self.test_dataset.ds_sizes)
            for i in range(self.num_scene):
                scene_mean_psnr = all_gather_ddp_if_available(scene_psnrs[i]).mean()
                # self.log(f'test/psnr/scene_{i}', scene_mean_psnr, True)
                self.logger.experiment.add_scalar('test/psnr/all', scene_mean_psnr, i)

        ssims = torch.stack([x['ssim'] for x in outputs])
        mean_ssim = all_gather_ddp_if_available(ssims).mean()
        self.log('test/ssim', mean_ssim)

        if self.hparams.eval_lpips:
            lpipss = torch.stack([x['lpips'] for x in outputs])
            mean_lpips = all_gather_ddp_if_available(lpipss).mean()
            self.log('test/lpips_vgg', mean_lpips)

    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items
    
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True

        if is_changed:
            checkpoint.pop("optimizer_states", None)


if __name__ == '__main__':
    hparams = get_opts()
    if not hparams.ckpt_path:
        raise ValueError('You need to provide a @ckpt_path for optimization!')

    system = NeRFSystem.load_from_checkpoint(hparams.ckpt_path, strict=False, hparams=hparams)

    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.dataset_name}/{hparams.exp_name}_opt',
                              filename='{epoch:d}',
                              save_weights_only=True,
                              every_n_epochs=hparams.num_epochs,
                              save_on_train_epoch_end=True,
                              save_top_k=-1)
    callbacks = [ckpt_cb, TQDMProgressBar(refresh_rate=1)]

    logger = TensorBoardLogger(save_dir=f"logs/{hparams.dataset_name}",
                               name=hparams.exp_name+'_opt',
                               default_hp_metric=False)

    trainer = Trainer(max_epochs=hparams.num_epochs if hparams.num_epochs else 1,
                      check_val_every_n_epoch=hparams.num_epochs if hparams.num_epochs else 1,
                      callbacks=callbacks,
                      logger=logger,
                      enable_model_summary=False,
                      accelerator='gpu',
                      devices=hparams.num_gpus,
                      strategy=DDPPlugin(find_unused_parameters=False)
                               if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=-1 if hparams.val_only else 0,
                      precision=16)

    trainer.fit(system)
    # trainer.fit(system, ckpt_path=hparams.ckpt_path)

    if not hparams.val_only: # save slimmed ckpt for the last epoch
        ckpt_ = \
            slim_ckpt(f'ckpts/{hparams.dataset_name}/{hparams.exp_name}_opt/epoch={hparams.num_epochs-1}.ckpt',
                      save_poses=hparams.optimize_ext)
        torch.save(ckpt_, f'ckpts/{hparams.dataset_name}/{hparams.exp_name}_opt/epoch={hparams.num_epochs-1}_slim.ckpt')

    if (not hparams.no_save_test) and \
       hparams.dataset_name=='nsvf' and \
       any('Synthetic' in s for s in hparams.root_dir): # save video
        imgs = sorted(glob.glob(os.path.join(system.val_dir, '*.png')))
        imageio.mimsave(os.path.join(system.val_dir, 'rgb.mp4'),
                        [imageio.imread(img) for img in imgs[::2]],
                        fps=30, macro_block_size=1)
        imageio.mimsave(os.path.join(system.val_dir, 'depth.mp4'),
                        [imageio.imread(img) for img in imgs[1::2]],
                        fps=30, macro_block_size=1)
