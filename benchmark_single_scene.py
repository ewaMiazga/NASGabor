import os
import json
import time
import argparse
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument("--dataset", required=True)
parser.add_argument("--output_root", required=True)
parser.add_argument(
    "--scene", required=True, help="Scene to run (e.g. 'bonsai' or 'garden')"
)
parser.add_argument(
    "--images", required=True, help="Image set to use (e.g. 'images_2' or 'images_4')"
)
parser.add_argument("--iterations", default="30000")

args = parser.parse_args()

CAP_MAX_BETA_SPLATTING = {
    "bicycle": 6_000_000,
    "flowers": 3_000_000,
    "garden": 5_000_000,
    "stump": 4_500_000,
    "treehill": 3_500_000,
    "room": 1_500_000,
    "counter": 1_500_000,
    "kitchen": 1_500_000,
    "bonsai": 1_500_000,
    "train": 1_000_000,
    "truck": 2_500_000,
    "drjohnson": 3_500_000,
    "playroom": 2_500_000,
    "chair": 300_000,
    "drums": 300_000,
    "ficus": 300_000,
    "hotdog": 300_000,
    "lego": 300_000,
    "materials": 300_000,
    "mic": 300_000,
    "ship": 300_000,
}

NERF_SYNTHETIC_SCENES = [
    "chair",
    "drums",
    "ficus",
    "hotdog",
    "lego",
    "materials",
    "mic",
    "ship",
]


def _infer_config_name(dataset_path: str, scene: str) -> str:
    ds = dataset_path.lower()
    for name in ("ns", "db", "tandt", "indoor", "outdoor"):
        if name in ds:
            return name
    if "nerf" in ds or "synthetic" in ds or scene in NERF_SYNTHETIC_SCENES:
        return "ns"
    if scene in {"room", "counter", "kitchen", "bonsai"}:
        return "indoor"
    if scene in {"bicycle", "flowers", "garden", "stump", "treehill"}:
        return "outdoor"
    if scene in {"train", "truck"}:
        return "tandt"
    if scene in {"drjohnson", "playroom"}:
        return "db"
    return "outdoor"


def _load_scene_config(dataset_path: str, scene: str) -> dict:
    config_name = _infer_config_name(dataset_path, scene)
    config_path = os.path.join(
        os.path.dirname(__file__), "configs", f"{config_name}.json"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    defaults = {
        "sh_lr": 0.00025,
        "gmm_featurea_lr": 0.0025,
        "opacity_lr": 0.05,
        "lobe_number": 1,
        "white_background": False,
    }
    defaults.update(cfg)
    return defaults


def get_storage_mb(scene_path, iteration):
    folder = os.path.join(
        scene_path, f"point_cloud/iteration_{iteration}/point_cloud.ply"
    )
    if os.path.exists(folder):
        return os.path.getsize(folder) / (1024 * 1024)


def run_scene(scene, images, cap_max):

    scene_path = os.path.join(args.dataset, scene)

    output = os.path.join(args.output_root, scene)

    os.makedirs(output, exist_ok=True)

    training_file = "train.py"

    lr_info = {}
    cfg = _load_scene_config(args.dataset, scene)
    cmd = [
        "python",
        training_file,
        "-s",
        scene_path,
        "--images",
        images,
        "-m",
        output,
        "--cap_max",
        str(cap_max),
        "--eval",
        "--disable_viewer",
        "--quiet",
        "--iterations",
        args.iterations,
        "--gmm_color_mode",
        "nasg_gabor",
        "--sh_degree",
        "0",
        "--lobe_number",
        str(cfg["lobe_number"]),
        "--auto_lr",
    ]

    if cfg.get("white_background", False):
        cmd.append("--white_background")

    print("\nRunning:", " ".join(cmd))

    start = time.time()
    subprocess.run(cmd, check=True)
    end = time.time()

    training_time = end - start

    metrics_file = os.path.join(
        output, f"point_cloud/iteration_{args.iterations}/metrics.json"
    )

    metrics = {}

    if os.path.exists(metrics_file):
        with open(metrics_file) as f:
            metrics = json.load(f)

    storage_MB = get_storage_mb(output, args.iterations)

    lr_report_path = os.path.join(output, "lr_report.json")
    if os.path.exists(lr_report_path):
        with open(lr_report_path) as f:
            lr_info = json.load(f)

    result = {
        "scene": scene,
        "mode": "nasg",
        "lobe_number": str(cfg["lobe_number"]),
        "training_time_sec": training_time,
        "storage_MB": storage_MB,
        **lr_info,
        **metrics,
    }

    with open(os.path.join(output, "experiment_results.json"), "w") as f:
        json.dump(result, f, indent=4)


run_scene(args.scene, args.images, CAP_MAX_BETA_SPLATTING.get(args.scene, 300_000))