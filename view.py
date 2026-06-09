import time
import torch
import viser
from argparse import ArgumentParser
from arguments import ModelParams, ViewerParams
from scene import BetaModel
from scene.beta_viewer import BetaViewer


@torch.no_grad()
def viewing(args):
    beta_model = BetaModel(args.lobe_number, args.gmm_color_mode)
    if args.ply:
        beta_model.load_ply(args.ply)
    else:
        raise ValueError("You must provide a .ply file (--ply)")

    bg_color = [1, 1, 1] if args.white_background else [0, 0, 0]
    beta_model.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    server = viser.ViserServer(port=args.port, verbose=False)
    viewer = BetaViewer(
        server=server,
        render_fn=lambda camera_state, render_tab_state: beta_model.view(
            camera_state, render_tab_state, args.center
        ),
        mode="rendering",
        share_url=args.share_url,
    )
    print("Viewer running... Ctrl+C to exit.")
    time.sleep(100000)


if __name__ == "__main__":
    parser = ArgumentParser(description="Viewing script parameters")
    ModelParams(parser), ViewerParams(parser)
    parser.add_argument("--ply", type=str, default=None, help="path to the .ply file")
    parser.add_argument(
        "--share_url", action="store_true", help="Share URL for the viewer"
    )
    parser.add_argument(
        "--center",
        action="store_true",
        help="Center the model in the viewer",
    )
    args = parser.parse_args()

    viewing(args)
