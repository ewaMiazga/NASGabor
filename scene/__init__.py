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

import os
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.beta_model import BetaModel
from arguments import ModelParams
from utils.camera_utils import (
    cameraList_from_camInfos,
    camera_to_JSON,
)
import torch
from utils.image_utils import psnr
from utils.camera_utils import mean_k_nn_distance, centers_from_cam_infos
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from fused_ssim import fused_ssim
from tqdm import tqdm
import numpy as np
from PIL import Image


class Scene:
    beta_model: BetaModel

    def __init__(
        self,
        args: ModelParams,
        beta_model: BetaModel,
        load_iteration=None,
        shuffle=True,
        resolution_scales=[1.0],
    ):
        self.model_path = args.model_path
        self.loaded_iter = None
        self.beta_model = beta_model
        self.best_psnr = 0

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](
                args.source_path, args.images, args.eval, init_type=args.init_type
            )
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](
                args.source_path, args.white_background, args.eval
            )
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(
                os.path.join(self.model_path, "input.ply"), "wb"
            ) as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras and not args.eval:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(
                scene_info.train_cameras
            )  # Multi-res consistent random shuffling
            random.shuffle(
                scene_info.test_cameras
            )  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.train_cameras, resolution_scale, args
            )
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(
                scene_info.test_cameras, resolution_scale, args
            )

        if self.loaded_iter:
            self.beta_model.load_ply(
                os.path.join(
                    self.model_path,
                    "point_cloud",
                    "iteration_" + str(self.loaded_iter),
                    "point_cloud.ply",
                )
            )
        else:
            self.beta_model.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

        metric_scene_info = scene_info
        if not args.eval:
            try:
                if os.path.exists(os.path.join(args.source_path, "sparse")):
                    metric_scene_info = sceneLoadTypeCallbacks["Colmap"](
                        args.source_path,
                        args.images,
                        True,
                        init_type=args.init_type,
                    )
                elif os.path.exists(
                    os.path.join(args.source_path, "transforms_train.json")
                ):
                    metric_scene_info = sceneLoadTypeCallbacks["Blender"](
                        args.source_path,
                        args.white_background,
                        True,
                    )
            except Exception:
                metric_scene_info = scene_info

        centers = centers_from_cam_infos(metric_scene_info.train_cameras)
        self.cam_mean_kdist = mean_k_nn_distance(centers, k=getattr(args, "cam_k", 3))

    def save(self, iteration):
        point_cloud_path = os.path.join(
            self.model_path, "point_cloud/iteration_{}".format(iteration)
        )
        self.beta_model.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]

    def getMeanDistance(self) -> float:
        return float(self.cam_mean_kdist)

    @torch.no_grad()
    def eval(self):
        torch.cuda.empty_cache()
        print("Starting evaluation...")
        psnr_test = 0.0
        ssim_test = 0.0
        lpips_test = 0.0
        fps_test = 0.0
        test_view_stack = self.getTestCameras()

        lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type="vgg", normalize=True
        ).to("cuda")

        features = self.beta_model._get_cuda_gmm_features()

        # Warmup: render a few frames to trigger any JIT compilation / kernel caching.
        N_WARMUP = 10
        for viewpoint in test_view_stack[:N_WARMUP]:
            self.beta_model.render_eval(viewpoint, activated_features=features)
        torch.cuda.synchronize()

        N_PASSES = 3
        print(
            f"Evaluating {N_PASSES} passes over test views for stable FPS measurement..."
        )
        for _ in range(N_PASSES):
            for idx, viewpoint in tqdm(enumerate(test_view_stack)):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize()
                s.record()
                self.beta_model.render_eval(viewpoint, activated_features=features)[
                    "render"
                ]

                e.record()
                torch.cuda.synchronize()
                end = s.elapsed_time(e)
                fps_test += 1 / end

        fps_test /= (
            len(test_view_stack) * N_PASSES
        )  # Average FPS per view across all passes
        fps_test *= 1000.0  # ms → seconds
        print("FPS evaluation completed.")

        for idx, viewpoint in tqdm(enumerate(test_view_stack)):
            image = torch.clamp(
                self.beta_model.render_eval(viewpoint, activated_features=features)[
                    "render"
                ],
                0.0,
                1.0,
            )

            # save image in render_test/
            os.makedirs(os.path.join(self.model_path, "render_test"), exist_ok=True)
            image_cpu = (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            image_cpu = Image.fromarray(image_cpu)
            image_cpu.save(
                os.path.join(self.model_path, "render_test", f"image_{idx}.png")
            )

            gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

            # save gt to gt_test/
            os.makedirs(os.path.join(self.model_path, "gt_test"), exist_ok=True)
            gt_image_cpu = (gt_image.permute(1, 2, 0).cpu().numpy() * 255).astype(
                np.uint8
            )
            gt_image_cpu = Image.fromarray(gt_image_cpu)
            gt_image_cpu.save(
                os.path.join(self.model_path, "gt_test", f"image_{idx}.png")
            )

            psnr_test += psnr(image, gt_image).mean()
            ssim_test += fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0)).mean()
            lpips_test += lpips_metric(image.unsqueeze(0), gt_image.unsqueeze(0)).item()
        psnr_test /= len(test_view_stack)
        ssim_test /= len(test_view_stack)
        lpips_test /= len(test_view_stack)

        result = {
            "SSIM": ssim_test.item(),
            "PSNR": psnr_test.item(),
            "LPIPS": lpips_test,
            "FPS": fps_test,
        }
        print(result)
        return result
