import os
import json
import torch
import sys
from scene import Scene, BetaModel
from argparse import ArgumentParser
from arguments import ModelParams


def training(args):
    beta_model = BetaModel(args.lobe_number, args.gmm_color_mode)
    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    beta_model.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    scene = Scene(args, beta_model)
    print("scene loaded ")
    ply_path = os.path.join(
        args.model_path, "point_cloud", "iteration_" + args.iteration, "point_cloud.ply"
    )
    print("ply_path:", ply_path)
    if os.path.exists(ply_path):
        print("Evaluating " + ply_path)
        beta_model.load_ply(ply_path)
        result = scene.eval()
        metrics_dir = os.path.join(
            args.model_path, "point_cloud", "iteration_" + args.iteration
        )
        os.makedirs(metrics_dir, exist_ok=True)
        metrics_path = os.path.join(metrics_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=1)
            f.write("\n")
        print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Evaluating script parameters")
    ModelParams(parser)
    parser.add_argument(
        "--iteration", default="30000", type=str, help="Iteration to evaluate"
    )
    args = parser.parse_args(sys.argv[1:])

    args.eval = True

    print("Evaluating " + args.model_path)

    training(args)
