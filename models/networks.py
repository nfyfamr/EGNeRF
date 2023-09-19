import torch
from torch import nn
from torch.nn.utils.stateless import functional_call
import tinycudann as tcnn
import vren
from einops import rearrange
import einops
from .custom_functions import TruncExp
import numpy as np

from .rendering import NEAR_DISTANCE


class NGP(nn.Module):
    def __init__(self, scale, hparams, rgb_act='Sigmoid'):
        super().__init__()

        self.rgb_act = rgb_act
        self.hyper = hparams.hyper
        self.embed_bias_feat = hparams.embed_bias_feat
        self.embed_bias_color = hparams.embed_bias_color
        self.scene_embed_mode = hparams.embed_mode # "sum" or "concat"
        self.scene_embed_size = hparams.embed_size
        self.num_scene = hparams.num_scene

        # scene bounding box
        self.scale = scale
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*scale)
        self.register_buffer('xyz_max', torch.ones(1, 3)*scale)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)

        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = max(1+int(np.ceil(np.log2(2*scale))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield',
            torch.ones(self.cascades*self.grid_size**3//8, dtype=torch.uint8) * 255)

        # constants
        L = hparams.L; F = hparams.F; log2_T = hparams.T; N_min = hparams.N_min; N_tables = hparams.N_tables
        b = np.exp(np.log(hparams.N_max*scale/N_min)/(L-1))    # default scale value is 0.5
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.feature_encoder = \
            tcnn.Encoding(
                n_input_dims=3, 
                encoding_config={
                    "otype": f"{hparams.grid}Grid",    # HashGrid / MixedFeatureGrid
                    "type": hparams.grid,     # Hash / MixedFeature
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "n_tables": N_tables,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                }
            )
        
        # A hypernetwork generating feature grids' parameters
        feat_modules = []
        feat_modules += [nn.Linear(self.scene_embed_size, hparams.fgen_channels), nn.ReLU()]
        for i in range(hparams.fgen_layers-2):
            feat_modules += [nn.Linear(hparams.fgen_channels, hparams.fgen_channels), nn.ReLU()]
        feat_modules += [nn.Linear(hparams.fgen_channels, self.feature_encoder.params.shape[0])]
        self.feature_generator = nn.Sequential(*feat_modules)

        # Scene embeddings
        print(f'# of scenes={self.num_scene}')
        self.register_parameter("scene_embed",
                                nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.num_scene, self.scene_embed_size, dtype=torch.float32)), True))
        density_net_input_dims = self.feature_encoder.n_output_dims + (self.scene_embed_size if self.scene_embed_mode == "concat" else 0)
        
        # Embedding bias
        if hparams.embed_bias_feat:
            self.register_parameter("scene_embed_bias_feat",
                                nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.num_scene, self.scene_embed_size, dtype=torch.float32)), True))
        if hparams.embed_bias_color:
            self.register_parameter("scene_embed_bias_color",
                                nn.Parameter(torch.nn.init.uniform_(torch.zeros(self.num_scene, self.scene_embed_size, dtype=torch.float32)), True))
        
        self.density_net = \
            tcnn.Network(
                n_input_dims=density_net_input_dims,
                n_output_dims=16,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        self.rgb_net = \
            tcnn.Network(
                n_input_dims=32, n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": self.rgb_act,
                    "n_neurons": hparams.rgb_channels,
                    "n_hidden_layers": hparams.rgb_layers,
                }
            )

        if self.rgb_act == 'None': # rgb_net output is log-radiance
            for i in range(3): # independent tonemappers for r,g,b
                tonemapper_net = \
                    tcnn.Network(
                        n_input_dims=1, n_output_dims=1,
                        network_config={
                            "otype": "FullyFusedMLP",
                            "activation": "ReLU",
                            "output_activation": "Sigmoid",
                            "n_neurons": 64,
                            "n_hidden_layers": 1,
                        }
                    )
                setattr(self, f'tonemapper_net_{i}', tonemapper_net)

    def density(self, x, embed=None, return_feat=False, embed_bias_feat=None, embed_bias_color=None):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            return_feat: whether to return intermediate feature

        Outputs:
            sigmas: (N)
        """
        x = (x-self.xyz_min)/(self.xyz_max-self.xyz_min)
        if self.hyper:
            if self.embed_bias_feat:
                p = {'params': self.feature_generator(embed+embed_bias_feat)}
            else:
                p = {'params': self.feature_generator(embed)}
            feat = functional_call(self.feature_encoder, p, x)
        else:
            feat = self.feature_encoder(x)
            
        # scene embedding
        if self.scene_embed_mode == "sum":
            if self.embed_bias_color:
                feat = feat + embed + embed_bias_color
            else:
                feat = feat + embed
        elif self.scene_embed_mode == "concat":
            if self.embed_bias_color:
                feat = torch.cat([feat, einops.repeat(embed+embed_bias_color, 'v -> n v', n=x.shape[0])], dim=1)
            else:
                feat = torch.cat([feat, einops.repeat(embed, 'v -> n v', n=x.shape[0])], dim=1)

        h = self.density_net(feat)

        sigmas = TruncExp.apply(h[:, 0])
        if return_feat: return sigmas, h
        return sigmas

    def log_radiance_to_rgb(self, log_radiances, **kwargs):
        """
        Convert log-radiance to rgb as the setting in HDR-NeRF.
        Called only when self.rgb_act == 'None' (with exposure)

        Inputs:
            log_radiances: (N, 3)

        Outputs:
            rgbs: (N, 3)
        """
        if 'exposure' in kwargs:
            log_exposure = torch.log(kwargs['exposure'])
        else: # unit exposure by default
            log_exposure = 0

        out = []
        for i in range(3):
            inp = log_radiances[:, i:i+1]+log_exposure
            out += [getattr(self, f'tonemapper_net_{i}')(inp)]
        rgbs = torch.cat(out, 1)
        return rgbs

    def forward(self, x, d, embed_idx, **kwargs):
        """
        Inputs:
            x: (N, 3) xyz in [-scale, scale]
            d: (N, 3) directions
            embed_idx: (N)

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        embed = self.scene_embed[embed_idx]
        embed_bias_feat = self.scene_embed_bias_feat[embed_idx] if self.embed_bias_feat else None
        embed_bias_color = self.scene_embed_bias_color[embed_idx] if self.embed_bias_color else None
        sigmas, h = self.density(x, embed, return_feat=True, embed_bias_feat=embed_bias_feat, embed_bias_color=embed_bias_color)
        d = d/torch.norm(d, dim=1, keepdim=True)
        d = self.dir_encoder((d+1)/2)
        rgbs = self.rgb_net(torch.cat([d, h], 1))

        if self.rgb_act == 'None': # rgbs is log-radiance
            if kwargs.get('output_radiance', False): # output HDR map
                rgbs = TruncExp.apply(rgbs)
            else: # convert to LDR using tonemapper networks
                rgbs = self.log_radiance_to_rgb(rgbs, **kwargs)

        return sigmas, rgbs

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold, scene=1):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>density_threshold)[:, 0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        self.count_grid = torch.zeros_like(self.density_grid)
        for scn_idx in range(self.num_scene):
            N_cams = poses[scn_idx].shape[0]
            w2c_R = rearrange(poses[scn_idx][:, :3, :3], 'n a b -> n b a') # (N_cams, 3, 3)
            w2c_T = -w2c_R@poses[scn_idx][:, :3, 3:] # (N_cams, 3, 1)
            cells = self.get_all_cells()
            for c in range(self.cascades):
                indices, coords = cells[c]
                for i in range(0, len(indices), chunk):
                    xyzs = coords[i:i+chunk]/(self.grid_size-1)*2-1
                    s = min(2**(c-1), self.scale)
                    half_grid_size = s/self.grid_size
                    xyzs_w = (xyzs*(s-half_grid_size)).T # (3, chunk)
                    xyzs_c = w2c_R @ xyzs_w + w2c_T # (N_cams, 3, chunk)
                    uvd = K[scn_idx] @ xyzs_c # (N_cams, 3, chunk)
                    uv = uvd[:, :2]/uvd[:, 2:] # (N_cams, 2, chunk)
                    in_image = (uvd[:, 2]>=0)& \
                            (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                            (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                    covered_by_cam = (uvd[:, 2]>=NEAR_DISTANCE)&in_image # (N_cams, chunk)
                    # if the cell is visible by at least one camera
                    self.count_grid[scn_idx, c, indices[i:i+chunk]] = \
                        count = covered_by_cam.sum(0)/N_cams

                    too_near_to_cam = (uvd[:, 2]<NEAR_DISTANCE)&in_image # (N, chunk)
                    # if the cell is too close (in front) to any camera
                    too_near_to_any_cam = too_near_to_cam.any(0)
                    # a valid cell should be visible by at least one camera and not too close to any camera
                    valid_mask = (count>0)&(~too_near_to_any_cam)
                    self.density_grid[scn_idx, c, indices[i:i+chunk]] = \
                        torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, scn_idx, density_threshold, warmup=False, decay=0.95, erode=False):
        N_dbits_per_scene = self.cascades*self.grid_size**3//8

        for scn_idx in range(self.num_scene):
            density_grid_tmp = torch.zeros_like(self.density_grid[scn_idx])
            if warmup: # during the first steps
                cells = self.get_all_cells()
            else:
                cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                            density_threshold,
                                                            scene=scn_idx)
            # infer sigmas
            for c in range(self.cascades):
                indices, coords = cells[c]
                s = min(2**(c-1), self.scale)
                half_grid_size = s/self.grid_size
                xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
                # pick random position in the cell by adding noise in [-hgs, hgs]
                xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
                embed_idxs = torch.zeros(xyzs_w.shape[0], dtype=torch.int, device=self.density_grid.device) + scn_idx
                embeds = self.scene_embed.index_select(0, embed_idxs)
                density_grid_tmp[c, indices] = self.density(xyzs_w, embeds)

            if erode:
                # My own logic. decay more the cells that are visible to few cameras
                decay = torch.clamp(decay**(1/self.count_grid[scn_idx]), 0.1, 0.95)
            self.density_grid[scn_idx] = \
                torch.where(self.density_grid[scn_idx]<0,
                            self.density_grid[scn_idx],
                            torch.maximum(self.density_grid[scn_idx]*decay, density_grid_tmp))

            mean_density = self.density_grid[scn_idx, self.density_grid[scn_idx]>0].mean().item()

            vren.packbits(self.density_grid[scn_idx], min(mean_density, density_threshold),
                        self.density_bitfield[scn_idx*N_dbits_per_scene:(scn_idx+1)*N_dbits_per_scene])