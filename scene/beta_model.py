#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, apply_depth_colormap
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from utils.spherical_utils import get_basis_parameterized
from sklearn.neighbors import NearestNeighbors
import math
import torch.nn.functional as F
from gsplat.rendering import rasterization
import json
from .beta_viewer import BetaRenderTabState

from arguments import PARAMS_SIZE, COLOR_SIZE, POSITION_SIZE, SHAPE_SIZE


def knn(x, K=4):
    x_np = x.cpu().numpy()
    model = NearestNeighbors(n_neighbors=K, metric="euclidean").fit(x_np)
    distances, _ = model.kneighbors(x_np)
    return torch.from_numpy(distances).to(x)


class BetaModel:
    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        def beta_activation(betas):
            return 4.0 * torch.exp(betas)

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize

        self.beta_activation = beta_activation

        self.color_func = COLOR_SIZE.get(self.gmm_color_mode)

        self.pos_activation = lambda x: torch.tanh(x)
        self.weight_activation = lambda x: torch.tanh(x)

    def __init__(self, lobe_number: int = 1, gmm_color_mode: str = "nasg_gabor"):
        self.assign_color_model(gmm_color_mode)
        self.active_lobe_number = 0
        self.max_lobe_number = lobe_number

        self._xyz = torch.empty(0)
        self._sh0 = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._beta = torch.empty(0)
        self.background = torch.empty(0)
        self.optimizer = None
        self.spatial_lr_scale = 0

        self.setup_functions()

    def capture(self):
        return (
            self.active_lobe_number,
            self._xyz,
            self._sh0,
            self._features_pos,
            self._features_shape,
            self._features_weight,
            self._scaling,
            self._rotation,
            self._opacity,
            self._beta,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def restore(self, model_args, training_args):
        (
            self.active_lobe_number,
            self._xyz,
            self._sh0,
            self._features_pos,
            self._features_shape,
            self._features_weight,
            self._scaling,
            self._rotation,
            self._opacity,
            self._beta,
            opt_dict,
            self.spatial_lr_scale,
        ) = model_args
        self.training_setup(training_args)
        self.optimizer.load_state_dict(opt_dict)

    def assign_color_model(self, gmm_color_mode):
        assert (
            gmm_color_mode in COLOR_SIZE
        ), f"Invalid GMM color mode: {gmm_color_mode}. Supported modes: {list(COLOR_SIZE.keys())}"
        self.gmm_color_mode = gmm_color_mode
        self.color_func = COLOR_SIZE.get(self.gmm_color_mode)
        self.position_size = POSITION_SIZE[self.gmm_color_mode]  # 3
        self.shape_size = SHAPE_SIZE[self.gmm_color_mode]
        self.color_size = COLOR_SIZE[self.gmm_color_mode]
        self.params_size = PARAMS_SIZE[
            self.gmm_color_mode
        ]  # total params per degree: position + shape + weight

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        return self._xyz

    @property
    def get_shs(self):
        sh0 = self._sh0
        shN = self._shN
        return torch.cat((sh0, shN), dim=1)

    @property
    def get_features_pos(self):
        return self.pos_activation(self._features_pos)

    @property
    def get_features_weight(self):
        return self.weight_activation(self._features_weight)

    @property
    def get_features_shape(self):
        return self._features_shape

    @property
    def get_features(self):
        sh0 = self._sh0

        # Interleaved layout: [pos, shape, weight] * lobes + base_color
        pos = self.get_features_pos.reshape(
            -1, self.max_lobe_number, self.position_size
        )
        shape = self._features_shape.reshape(-1, self.max_lobe_number, self.shape_size)
        weight = self.get_features_weight.reshape(
            -1, self.max_lobe_number, self.color_size
        )

        # Interleave: [pos, shape, weight] for each lobe
        interleaved = torch.cat([pos, shape, weight], dim=2)

        flattened = interleaved.reshape(
            -1, 1, self.max_lobe_number * self.params_size
        )  # [N, 1, max_lobe_number * params_size]

        return torch.cat(
            (sh0.reshape(-1, 1, 3), flattened), dim=2
        )  # [N, 1, total_params_per_point]

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_beta(self):
        return self.beta_activation(self._beta)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(
            self.get_scaling, scaling_modifier, self._rotation
        )

    def oneupSHdegree(self):
        if self.active_lobe_number < self.max_lobe_number:
            self.active_lobe_number += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()

        fused_color = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        # Allocate tensor
        features = (
            torch.empty(
                (fused_color.shape[0], 1, (self.max_lobe_number) * self.params_size + 3)
            )
            .float()
            .cuda()
        )

        for i in range(self.max_lobe_number):
            start = i * self.params_size
            # positions
            features[:, 0, start : start + self.position_size] = 0.0
            # shape
            features[
                :,
                0,
                start
                + self.position_size : start
                + self.position_size
                + self.shape_size,
            ] = math.log(0.5)

            # if nasg gabor, initialize k to -1.6 for tanh *20
            if self.gmm_color_mode == "nasg_gabor":
                k_start = start + self.position_size + self.shape_size - 1
                k_end = start + self.position_size + self.shape_size
                features[:, 0, k_start:k_end] = (
                    -1.6
                )  # Initialize k parameters to -1.6 -> after activation: 1.56
            # weight
            features[
                :,
                0,
                start
                + self.position_size
                + self.shape_size : start
                + self.position_size
                + self.shape_size
                + self.color_size,
            ] = 0.5

        # Finally, append base color at the very end
        features[:, 0, -3:] = fused_color

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = (
            knn(torch.from_numpy(np.asarray(pcd.points)).float().cuda())[:, 1:] ** 2
        ).mean(dim=-1)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(
            0.5
            * torch.ones(
                (fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"
            )
        )

        ## Betas initialization
        betas = torch.zeros_like(opacities)
        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))

        self._sh0 = nn.Parameter(
            features[:, :, -3:].transpose(1, 2).contiguous().requires_grad_(True)
        )

        # Extract interleaved params: [pos, shape, weight] * max_lobe_number
        params_only = features[:, 0, :-3]  # [N, max_lobe_number * params_size]
        params_reshaped = params_only.reshape(
            features.shape[0], self.max_lobe_number, self.params_size
        )  # [N, max_lobe_number, params_size]

        # Extract each component type
        pos = params_reshaped[:, :, 0 : self.position_size]
        shape = params_reshaped[
            :, :, self.position_size : self.position_size + self.shape_size
        ]
        weight = params_reshaped[:, :, self.position_size + self.shape_size :]

        # Reshape and store as parameters
        self._features_pos = nn.Parameter(
            pos.reshape(features.shape[0], self.max_lobe_number * self.position_size)
            .unsqueeze(1)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_shape = nn.Parameter(
            shape.reshape(features.shape[0], self.max_lobe_number * self.shape_size)
            .unsqueeze(1)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_weight = nn.Parameter(
            weight.reshape(features.shape[0], self.max_lobe_number * self.color_size)
            .unsqueeze(1)
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )

        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self._beta = nn.Parameter(betas.requires_grad_(True))

    def prune(self, live_mask):
        self._xyz = self._xyz[live_mask]
        self._sh0 = self._sh0[live_mask]
        self._features_pos = self._features_pos[live_mask]
        self._features_shape = self._features_shape[live_mask]
        self._features_weight = self._features_weight[live_mask]
        self._scaling = self._scaling[live_mask]
        self._rotation = self._rotation[live_mask]
        self._opacity = self._opacity[live_mask]
        self._beta = self._beta[live_mask]

    def compute_lr(
        self,
        d_nn,
        d_nn_ref=0.3536,
        feature_base_lr=0.0025,
        opacity_base_lr=0.025,
        min_opacity_lr=0.025,
        max_opacity_lr=0.05,
    ):
        main_lr = feature_base_lr * (d_nn_ref / d_nn) ** 2.0

        opacity_lr = max(
            min_opacity_lr,
            min(max_opacity_lr, opacity_base_lr * (d_nn / d_nn_ref) ** 0.6),
        )

        return float(main_lr), float(opacity_lr)

    def training_setup(self, training_args):
        self.sh_lr = 0.00025
        if getattr(training_args, "auto_lr", False):

            f_lr, op_lr = self.compute_lr(training_args.scene_dnn_mean)
            self.gmm_features_lr = f_lr
            self.opacity_lr = op_lr
        else:
            # for constant lr from config
            self.gmm_features_lr = training_args.gmm_features_lr
            self.opacity_lr = training_args.opacity_lr

        # save lr to lr_report.json
        lr_report = {
            "sh_lr": self.sh_lr,
            "gmm_features_lr": self.gmm_features_lr,
            "opacity_lr": self.opacity_lr,
        }

        with open(os.path.join(training_args.model_path, "lr_report.json"), "w") as f:
            json.dump(lr_report, f, indent=4)

        l = [
            {
                "params": [self._xyz],
                "lr": training_args.position_lr_init * self.spatial_lr_scale,
                "name": "xyz",
            },
            {"params": [self._sh0], "lr": self.sh_lr, "name": "sh0"},
            {
                "params": [self._features_pos],
                "lr": self.gmm_features_lr,
                "name": "features_pos",
            },
            {
                "params": [self._features_shape],
                "lr": self.gmm_features_lr,
                "name": "features_shape",
            },
            {
                "params": [self._features_weight],
                "lr": self.gmm_features_lr,
                "name": "features_weight",
            },
            {
                "params": [self._opacity],
                "lr": self.opacity_lr,
                "name": "opacity",
            },
            {
                "params": [self._beta],
                "lr": training_args.beta_lr,
                "name": "beta",
            },
            {
                "params": [self._scaling],
                "lr": training_args.scaling_lr,
                "name": "scaling",
            },
            {
                "params": [self._rotation],
                "lr": training_args.rotation_lr,
                "name": "rotation",
            },
        ]

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init * self.spatial_lr_scale,
            lr_final=training_args.position_lr_final * self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps,
        )

        def lr_lambda(step):
            T_max = training_args.iterations
            return 0.5 * (1 + math.cos(math.pi * step / T_max))  # cosine decay

        not_schedule_params = ["xyz"]

        # Create scheduler that only adjusts specific groups
        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=[
                (
                    lambda step: (
                        1.0 if pg["name"] in not_schedule_params else lr_lambda(step)
                    )
                )
                for pg in self.optimizer.param_groups
            ],
        )

    def update_learning_rate(self, iteration):
        """Learning rate scheduling per step"""
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group["lr"] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ["x", "y", "z", "nx", "ny", "nz"]
        # All channels except the 3 DC
        for i in range(self._sh0.shape[1] * self._sh0.shape[2]):
            l.append("sh0_{}".format(i))

        for i in range(
            self._features_pos.shape[1] * self._features_pos.shape[2]
            + self._features_shape.shape[1] * self._features_shape.shape[2]
            + self._features_weight.shape[1] * self._features_weight.shape[2]
        ):
            l.append("f_rest_{}".format(i))
        l.append("opacity")
        l.append("beta")
        for i in range(self._scaling.shape[1]):
            l.append("scale_{}".format(i))
        for i in range(self._rotation.shape[1]):
            l.append("rot_{}".format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        sh0 = self._sh0.detach().flatten(start_dim=1).contiguous().cpu().numpy()
        # Interleaved layout: [pos, shape, weight] * lobes
        pos = self._features_pos.squeeze(-1).reshape(
            -1, self.max_lobe_number, self.position_size
        )
        shape = self._features_shape.squeeze(-1).reshape(
            -1, self.max_lobe_number, self.shape_size
        )
        weight = self._features_weight.squeeze(-1).reshape(
            -1, self.max_lobe_number, self.color_size
        )
        interleaved = torch.cat(
            [pos, shape, weight], dim=2
        )  # [N, max_lobe_number, params_size]
        gmm_params = (
            interleaved.reshape(-1, self.max_lobe_number * self.params_size)
            .detach()
            .cpu()
            .numpy()
        )

        opacities = self._opacity.detach().cpu().numpy()
        betas = self._beta.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [
            (attribute, "f4") for attribute in self.construct_list_of_attributes()
        ]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate(
            (xyz, normals, sh0, gmm_params, opacities, betas, scale, rotation),
            axis=1,
        )
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, "vertex")
        PlyData([el]).write(path)

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack(
            (
                np.asarray(plydata.elements[0]["x"]),
                np.asarray(plydata.elements[0]["y"]),
                np.asarray(plydata.elements[0]["z"]),
            ),
            axis=1,
        )
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]
        betas = np.asarray(plydata.elements[0]["beta"])[..., np.newaxis]

        sh0 = np.zeros((xyz.shape[0], 1, 3))
        sh0[:, 0, 0] = np.asarray(plydata.elements[0]["sh0_0"])
        sh0[:, 0, 1] = np.asarray(plydata.elements[0]["sh0_1"])
        sh0[:, 0, 2] = np.asarray(plydata.elements[0]["sh0_2"])

        extra_f_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("f_rest_")
        ]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split("_")[-1]))
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))

        if features_extra.shape[1] % 8 == 0:
            self.assign_color_model("nasg")
            assert (
                features_extra.shape[1] % self.params_size == 0
            ), f"Number of feature channels in ply does not match expected params size. {self.params_size}"
        else:
            self.assign_color_model("nasg_gabor")
            assert (
                features_extra.shape[1] % self.params_size == 0
            ), f"Number of feature channels in ply does not match expected params size. {self.params_size}"
        self.max_lobe_number = features_extra.shape[1] // self.params_size

        print("Number of points loaded from ply: ", xyz.shape[0])

        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape to [N, 1, max_lobe_number * params_size]
        features_extra = features_extra.reshape(
            (features_extra.shape[0], 1, (self.max_lobe_number) * self.params_size)
        )

        scale_names = [
            p.name
            for p in plydata.elements[0].properties
            if p.name.startswith("scale_")
        ]
        scale_names = sorted(scale_names, key=lambda x: int(x.split("_")[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [
            p.name for p in plydata.elements[0].properties if p.name.startswith("rot")
        ]
        rot_names = sorted(rot_names, key=lambda x: int(x.split("_")[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(
            torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._sh0 = nn.Parameter(
            torch.tensor(sh0, dtype=torch.float, device="cuda")
            .transpose(1, 2)
            .contiguous()
            .requires_grad_(True)
        )

        # Interleaved format: [N, 1, max_lobe_number * params_size]
        # Reshape and extract components
        params_reshaped = features_extra.reshape(
            -1, self.max_lobe_number, self.params_size
        )
        pos_np = params_reshaped[:, :, 0 : self.position_size]
        shape_np = params_reshaped[
            :, :, self.position_size : self.position_size + self.shape_size
        ]
        weight_np = params_reshaped[:, :, self.position_size + self.shape_size :]

        self._features_pos = nn.Parameter(
            torch.tensor(
                pos_np.reshape(-1, self.max_lobe_number * self.position_size),
                dtype=torch.float,
                device="cuda",
            )
            .unsqueeze(-1)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_shape = nn.Parameter(
            torch.tensor(
                shape_np.reshape(-1, self.max_lobe_number * self.shape_size),
                dtype=torch.float,
                device="cuda",
            )
            .unsqueeze(-1)
            .contiguous()
            .requires_grad_(True)
        )
        self._features_weight = nn.Parameter(
            torch.tensor(
                weight_np.reshape(-1, self.max_lobe_number * self.color_size),
                dtype=torch.float,
                device="cuda",
            )
            .unsqueeze(-1)
            .contiguous()
            .requires_grad_(True)
        )
        self._opacity = nn.Parameter(
            torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(
                True
            )
        )
        self._beta = nn.Parameter(
            torch.tensor(betas, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._scaling = nn.Parameter(
            torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True)
        )
        self._rotation = nn.Parameter(
            torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True)
        )

        self.active_lobe_number = self.max_lobe_number

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group["params"][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    (group["params"][0][mask].requires_grad_(True))
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    group["params"][0][mask].requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat(
                    (stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0
                )
                stored_state["exp_avg_sq"] = torch.cat(
                    (stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)),
                    dim=0,
                )

                del self.optimizer.state[group["params"][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                self.optimizer.state[group["params"][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat(
                        (group["params"][0], extension_tensor), dim=0
                    ).requires_grad_(True)
                )
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(
        self,
        new_xyz,
        new_sh0,
        new_features_pos,
        new_features_shape,
        new_features_weight,
        new_opacities,
        new_betas,
        new_scaling,
        new_rotation,
        reset_params=True,
    ):
        d = {
            "xyz": new_xyz,
            "sh0": new_sh0,
            "features_pos": new_features_pos,
            "features_shape": new_features_shape,
            "features_weight": new_features_weight,
            "opacity": new_opacities,
            "beta": new_betas,
            "scaling": new_scaling,
            "rotation": new_rotation,
        }

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._sh0 = optimizable_tensors["sh0"]
        self._features_pos = optimizable_tensors["features_pos"]
        self._features_shape = optimizable_tensors["features_shape"]
        self._features_weight = optimizable_tensors["features_weight"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

    def replace_tensors_to_optimizer(self, inds=None):
        tensors_dict = {
            "xyz": self._xyz,
            "sh0": self._sh0,
            "features_pos": self._features_pos,
            "features_shape": self._features_shape,
            "features_weight": self._features_weight,
            "opacity": self._opacity,
            "beta": self._beta,
            "scaling": self._scaling,
            "rotation": self._rotation,
        }

        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            tensor = tensors_dict[group["name"]]

            if tensor.numel() == 0:
                optimizable_tensors[group["name"]] = group["params"][0]
                continue

            stored_state = self.optimizer.state.get(group["params"][0], None)

            if inds is not None:
                stored_state["exp_avg"][inds] = 0
                stored_state["exp_avg_sq"][inds] = 0
            else:
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

            del self.optimizer.state[group["params"][0]]
            group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
            self.optimizer.state[group["params"][0]] = stored_state

            optimizable_tensors[group["name"]] = group["params"][0]

        self._xyz = optimizable_tensors["xyz"]
        self._sh0 = optimizable_tensors["sh0"]
        self._features_pos = optimizable_tensors["features_pos"]
        self._features_shape = optimizable_tensors["features_shape"]
        self._features_weight = optimizable_tensors["features_weight"]
        self._opacity = optimizable_tensors["opacity"]
        self._beta = optimizable_tensors["beta"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        torch.cuda.empty_cache()

        return optimizable_tensors

    def _update_params(self, idxs, ratio):
        new_opacity = 1.0 - torch.pow(
            1.0 - self.get_opacity[idxs, 0], 1.0 / (ratio + 1)
        )
        new_opacity = torch.clamp(
            new_opacity.unsqueeze(-1),
            max=1.0 - torch.finfo(torch.float32).eps,
            min=0.005,
        )
        new_opacity = self.inverse_opacity_activation(new_opacity)
        return (
            self._xyz[idxs],
            self._sh0[idxs],
            self._features_pos[idxs],
            self._features_shape[idxs],
            self._features_weight[idxs],
            new_opacity,
            self._beta[idxs],
            self._scaling[idxs],
            self._rotation[idxs],
        )

    def _get_background_rgb(self, device):
        if self.background.numel() == 3:
            return self.background.to(device=device)
        return torch.zeros(3, device=device)

    def _get_cuda_gmm_features(self):
        sh0 = self._sh0.reshape(-1, 3)
        pos = self.get_features_pos.squeeze(-1).reshape(
            -1, self.max_lobe_number, self.position_size
        )
        shape = self._features_shape.squeeze(-1).reshape(
            -1, self.max_lobe_number, self.shape_size
        )
        weight = self.get_features_weight.squeeze(-1).reshape(
            -1, self.max_lobe_number, self.color_size
        )

        clamp_min, clamp_max = -0.999999, 0.999999
        cos_theta = torch.clamp(pos[..., 0], clamp_min, clamp_max)
        cos_phi = torch.clamp(pos[..., 1], clamp_min, clamp_max)
        cos_tau = torch.clamp(pos[..., 2], clamp_min, clamp_max)

        x_basis, z_basis = get_basis_parameterized(cos_theta, cos_phi, cos_tau)

        # [lambda, a, k] activation in PyTorch.
        shape[..., 0] = torch.clamp(
            torch.exp(shape[..., 0]),
            max=1e4,
        )
        shape[..., 1] = torch.clamp(
            torch.exp(shape[..., 1]),
            max=1e4,
        )
        if self.gmm_color_mode == "nasg_gabor":
            shape[..., 2] = torch.clamp(
                (torch.tanh(shape[..., 2]) + 1.0) * 20.0, min=0.0, max=40.0
            )

        # For CUDA NASG-Gabor: pack frame explicitly as [x(3), z(3), lambda, a, k, rgb(3)]
        interleaved = torch.cat([x_basis, z_basis, shape, weight], dim=2).reshape(
            -1, self.max_lobe_number * (self.params_size + 3)
        )
        return torch.cat((sh0, interleaved), dim=1)

    def _sample_alives(self, probs, num, alive_indices=None):
        probs = probs / (probs.sum() + torch.finfo(torch.float32).eps)
        sampled_idxs = torch.multinomial(probs, num, replacement=True)
        if alive_indices is not None:
            sampled_idxs = alive_indices[sampled_idxs]
        ratio = torch.bincount(sampled_idxs)[sampled_idxs]
        return sampled_idxs, ratio

    def relocate_gs(self, dead_mask=None):
        if dead_mask.sum() == 0:
            return
        alive_mask = ~dead_mask
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        alive_indices = alive_mask.nonzero(as_tuple=True)[0]

        if alive_indices.shape[0] <= 0:
            return

        # sample from alive ones based on opacity
        probs = self.get_opacity[alive_indices, 0]
        reinit_idx, ratio = self._sample_alives(
            alive_indices=alive_indices, probs=probs, num=dead_indices.shape[0]
        )

        (
            self._xyz[dead_indices],
            self._sh0[dead_indices],
            self._features_pos[dead_indices],
            self._features_shape[dead_indices],
            self._features_weight[dead_indices],
            self._opacity[dead_indices],
            self._beta[dead_indices],
            self._scaling[dead_indices],
            self._rotation[dead_indices],
        ) = self._update_params(reinit_idx, ratio=ratio)

        self._opacity[reinit_idx] = self._opacity[dead_indices]

        self.replace_tensors_to_optimizer(inds=reinit_idx)

    def add_new_gs(self, cap_max):
        current_num_points = self._opacity.shape[0]
        target_num = min(cap_max, int(1.05 * current_num_points))
        num_gs = max(0, target_num - current_num_points)
        if num_gs <= 0:
            return 0

        probs = self.get_opacity.squeeze(-1)
        add_idx, ratio = self._sample_alives(probs=probs, num=num_gs)

        (
            new_xyz,
            new_sh0,
            new_features_pos,
            new_features_shape,
            new_features_weight,
            new_opacity,
            new_beta,
            new_scaling,
            new_rotation,
        ) = self._update_params(add_idx, ratio=ratio)

        self._opacity[add_idx] = new_opacity

        self.densification_postfix(
            new_xyz,
            new_sh0,
            new_features_pos,
            new_features_shape,
            new_features_weight,
            new_opacity,
            new_beta,
            new_scaling,
            new_rotation,
            reset_params=False,
        )

        self.replace_tensors_to_optimizer(inds=add_idx)

        return num_gs

    def render(self, viewpoint_camera, render_mode="RGB", mask=None):
        if mask == None:
            mask = torch.ones_like(self.get_beta.squeeze()).bool()

        K = torch.zeros((3, 3), device=viewpoint_camera.projection_matrix.device)

        fx = 0.5 * viewpoint_camera.image_width / math.tan(viewpoint_camera.FoVx / 2)
        fy = 0.5 * viewpoint_camera.image_height / math.tan(viewpoint_camera.FoVy / 2)

        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = viewpoint_camera.image_width / 2
        K[1, 2] = viewpoint_camera.image_height / 2
        K[2, 2] = 1.0

        features = self.get_features.view(
            -1, (self.max_lobe_number) * self.params_size + 3
        )
        colors = torch.zeros((features.shape[0], 3), device=features.device)
        rgbs, alphas, meta = rasterization(
            means=self.get_xyz[mask],
            quats=self.get_rotation[mask],
            scales=self.get_scaling[mask],
            opacities=self.get_opacity.squeeze()[mask],
            betas=self.get_beta.squeeze()[mask],
            gmm_mode=self.gmm_color_mode,
            colors=colors,
            viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=viewpoint_camera.image_width,
            height=viewpoint_camera.image_height,
            backgrounds=self.background.unsqueeze(0),
            render_mode=render_mode,
            covars=None,
            sh_degree=0,
            lobe_number=self.active_lobe_number,
            gmm_features=features,
            packed=False,
        )

        # # Convert from N,H,W,C to N,C,H,W format
        rgbs = rgbs.permute(0, 3, 1, 2).contiguous()[0]

        return {
            "render": rgbs,
            "viewspace_points": meta["means2d"],
            "visibility_filter": meta["radii"] > 0,
            "radii": meta["radii"],
            "is_used": meta["radii"] > 0,
        }

    def render_eval(
        self, viewpoint_camera, render_mode="RGB", mask=None, activated_features=None
    ):
        if mask == None:
            mask = torch.ones_like(self.get_beta.squeeze()).bool()

        K = torch.zeros((3, 3), device=viewpoint_camera.projection_matrix.device)

        fx = 0.5 * viewpoint_camera.image_width / math.tan(viewpoint_camera.FoVx / 2)
        fy = 0.5 * viewpoint_camera.image_height / math.tan(viewpoint_camera.FoVy / 2)

        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = viewpoint_camera.image_width / 2
        K[1, 2] = viewpoint_camera.image_height / 2
        K[2, 2] = 1.0

        colors = torch.zeros(
            (activated_features.shape[0], 3), device=activated_features.device
        )

        rgbs, alphas, meta = rasterization(
            means=self.get_xyz[mask],
            quats=self.get_rotation[mask],
            scales=self.get_scaling[mask],
            opacities=self.get_opacity.squeeze()[mask],
            betas=self.get_beta.squeeze()[mask],
            gmm_mode=self.gmm_color_mode,
            colors=colors,
            viewmats=viewpoint_camera.world_view_transform.transpose(0, 1).unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=viewpoint_camera.image_width,
            height=viewpoint_camera.image_height,
            backgrounds=self.background.unsqueeze(0),
            render_mode=render_mode,
            covars=None,
            lobe_number=self.active_lobe_number,
            gmm_features=activated_features,
            packed=False,
            eval=True,
        )

        # # Convert from N,H,W,C to N,C,H,W format
        rgbs = rgbs.permute(0, 3, 1, 2).contiguous()[0]

        return {
            "render": rgbs,
            "viewspace_points": meta["means2d"],
            "visibility_filter": meta["radii"] > 0,
            "radii": meta["radii"],
            "is_used": meta["radii"] > 0,
        }

    @torch.no_grad()
    def view(self, camera_state, render_tab_state, center=None):
        """Callable function for the viewer."""
        assert isinstance(render_tab_state, BetaRenderTabState)
        if render_tab_state.preview_render:
            W = render_tab_state.render_width
            H = render_tab_state.render_height
        else:
            W = render_tab_state.viewer_width
            H = render_tab_state.viewer_height
        c2w = camera_state.c2w
        K = camera_state.get_K((W, H))
        c2w = torch.from_numpy(c2w).float().to("cuda")
        K = torch.from_numpy(K).float().to("cuda")

        if center:
            xyz = self._xyz - self._xyz.mean(dim=0, keepdim=True)
        else:
            xyz = self._xyz

        render_mode = render_tab_state.render_mode
        mask = torch.logical_and(
            self._beta >= render_tab_state.b_range[0],
            self._beta <= render_tab_state.b_range[1],
        ).squeeze()
        self.background = (
            torch.tensor(render_tab_state.backgrounds, device="cuda") / 255.0
        )

        features = self.get_features.view(
            -1, (self.max_lobe_number) * self.params_size + 3
        )
        colors = torch.zeros((features.shape[0], 3), device=features.device)
        render_colors, alphas, meta = rasterization(
            means=xyz[mask],
            quats=self.get_rotation[mask],
            scales=self.get_scaling[mask],
            opacities=self.get_opacity.squeeze()[mask],
            betas=self.get_beta.squeeze()[mask],
            gmm_mode=self.gmm_color_mode,
            colors=colors,
            viewmats=torch.linalg.inv(c2w).unsqueeze(0),
            Ks=K.unsqueeze(0),
            width=W,
            height=H,
            backgrounds=self.background.unsqueeze(0),
            render_mode=render_mode if render_mode != "Alpha" else "RGB",
            covars=None,
            lobe_number=self.active_lobe_number,
            gmm_features=features,
            packed=False,
            near_plane=render_tab_state.near_plane,
            far_plane=render_tab_state.far_plane,
            radius_clip=render_tab_state.radius_clip,
        )

        render_tab_state.total_count_number = len(self.get_xyz)
        render_tab_state.rendered_count_number = (meta["radii"] > 0).sum().item()

        if render_mode == "Alpha":
            render_colors = alphas

        if render_colors.shape[-1] == 1:
            render_colors = apply_depth_colormap(render_colors)

        return render_colors[0].cpu().numpy()
