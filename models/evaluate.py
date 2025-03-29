from pathlib import Path
import torch
from torchvision.transforms import ToPILImage

from util.general_utils import save_video
from util.utils import save_depth_map
from omegaconf import OmegaConf


def evaluate(model):
    fps = model.config["save_fps"]
    save_root = Path(model.run_dir)

    video = (255 * torch.cat(model.images, dim=0)).to(torch.uint8).detach().cpu()
    video_reverse = (
        (255 * torch.cat(model.images[::-1], dim=0)).to(torch.uint8).detach().cpu()
    )

    save_video(video, save_root / "output.mp4", fps=fps)
    save_video(video_reverse, save_root / "output_reverse.mp4", fps=fps)


def evaluate_epoch(model, epoch, vmax=None):
    rendered_depth = model.rendered_depths[epoch].clamp(0).cpu().numpy()
    depth = model.depths[epoch].clamp(0).cpu().numpy()
    save_root = Path(model.run_dir) / "images"
    save_root.mkdir(exist_ok=True, parents=True)
    (save_root / "inpaint_input_image").mkdir(exist_ok=True, parents=True)
    (save_root / "frames").mkdir(exist_ok=True, parents=True)
    (save_root / "masks").mkdir(exist_ok=True, parents=True)
    (save_root / "post_masks").mkdir(exist_ok=True, parents=True)
    (save_root / "rendered_images").mkdir(exist_ok=True, parents=True)
    (save_root / "rendered_depths").mkdir(exist_ok=True, parents=True)
    (save_root / "depth").mkdir(exist_ok=True, parents=True)

    model.inpaint_input_image[epoch].save(
        save_root / "inpaint_input_image" / f"{epoch}.png"
    )
    ToPILImage()(model.images[epoch][0]).save(save_root / "frames" / f"{epoch}.png")
    ToPILImage()(model.masks[epoch][0]).save(save_root / "masks" / f"{epoch}.png")
    ToPILImage()(model.post_masks[epoch][0]).save(
        save_root / "post_masks" / f"{epoch}.png"
    )
    ToPILImage()(model.rendered_images[epoch][0]).save(
        save_root / "rendered_images" / f"{epoch}.png"
    )
    save_depth_map(
        rendered_depth, save_root / "rendered_depths" / f"{epoch}.png", vmax=vmax
    )
    save_depth_map(
        depth, save_root / "depth" / f"{epoch}.png", vmax=vmax, save_clean=True
    )

    if hasattr(model, "outter_masks"):
        (save_root / "outter_masks").mkdir(exist_ok=True, parents=True)
        ToPILImage()(model.outter_masks[epoch]).save(
            save_root / "outter_masks" / f"{epoch}.png"
        )
    if epoch == 0:
        with open(Path(model.run_dir) / "config.yaml", "w") as f:
            OmegaConf.save(model.config, f)
