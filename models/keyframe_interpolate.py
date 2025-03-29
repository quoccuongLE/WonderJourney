from copy import deepcopy
from pathlib import Path

import torch
from torch.nn import functional as F
from tqdm import tqdm

from models.evaluate import evaluate, evaluate_epoch
from models.models import KeyframeInterp
from util.finetune_utils import finetune_decoder
from util.utils import empty_cache


def keyframe_interpolate(
    scene_id: int,
    keyframe_id: int,
    config,
    kf_gen_dict,
    inpainting_prompt,
    inpainter_pipeline,
    vae,
    vmax,
    cutoff_depth,
    rotation_path,
):

    rotation = kf_gen_dict["rotation"]
    kf1_image = kf_gen_dict["kf1_image"]
    kf2_image = kf_gen_dict["kf2_image"]
    kf1_depth = kf_gen_dict["kf1_depth"]
    kf2_depth = kf_gen_dict["kf2_depth"]
    kf1_camera = kf_gen_dict["kf1_camera"]
    kf2_camera = kf_gen_dict["kf2_camera"]
    kf2_mask = kf_gen_dict["kf2_mask"]
    adaptive_negative_prompt = kf_gen_dict["adaptive_negative_prompt"]
    rotation = kf_gen_dict["rotation"]

    is_last_scene = scene_id == config['num_scenes'] - 1
    is_last_keyframe = keyframe_id == config['num_keyframes'] - 1
    try:
        is_next_rotation = rotation_path[scene_id*config['num_keyframes'] + keyframe_id + 1] != 0
    except IndexError:
        is_next_rotation = False
    try:
        is_previous_rotation = rotation_path[scene_id*config['num_keyframes'] + keyframe_id - 1] != 0
    except IndexError:
        is_previous_rotation = False
    is_beginning = scene_id == 0 and keyframe_id == 0
    speed_up = (rotation == 0) and ((is_last_scene and is_last_keyframe) or is_next_rotation)
    speed_down = (rotation == 0) and (is_beginning or is_previous_rotation)
    total_frames = config["frames"]
    total_frames = total_frames + config["frames"] // 5 if speed_up else total_frames
    total_frames = total_frames + config["frames"] // 5 if speed_down else total_frames
    kf_interp = KeyframeInterp(config, inpainter_pipeline, None, vae, rotation, 
                            F.to_pil_image(kf1_image[0]), inpainting_prompt, adaptive_negative_prompt,
                            kf2_upsample_coef=config['kf2_upsample_coef'], kf1_image=kf1_image, kf2_image=kf2_image,
                            kf1_depth=kf1_depth, kf2_depth=kf2_depth, kf1_camera=kf1_camera, kf2_camera=kf2_camera, kf2_mask=kf2_mask,
                            speed_up=speed_up, speed_down=speed_down, total_frames=total_frames
                            ).to(config["device"])
    save_root = Path(kf_interp.run_dir) / "images"
    save_root.mkdir(exist_ok=True, parents=True)
    F.to_pil_image(kf1_image[0]).save(save_root / "kf1.png")
    F.to_pil_image(kf2_image[0]).save(save_root / "kf2.png")

    kf2_camera_upsample, kf2_depth_upsample, kf2_mask_upsample, kf2_image_upsample = kf_interp.upsample_kf2()

    kf_interp.update_additional_point_cloud(kf2_depth_upsample, kf2_image_upsample, valid_mask=kf2_mask_upsample, camera=kf2_camera_upsample, points_2d=kf_interp.points_kf2)
    inconsistent_additional_point_index = kf_interp.visibility_check()
    kf2_depth_updated = kf_interp.update_additional_point_depth(inconsistent_additional_point_index, depth=kf2_depth_upsample, mask=kf2_mask_upsample)
    # save_depth_map(kf2_depth_updated.detach().cpu().numpy(), save_root / 'kf2_depth_updated', vmin=0, vmax=vmax)
    kf_interp.reset_additional_point_cloud()
    kf_interp.update_additional_point_cloud(kf2_depth_updated, kf2_image_upsample, valid_mask=kf2_mask_upsample, camera=kf2_camera_upsample, points_2d=kf_interp.points_kf2)

    kf_interp.depths[0] = F.interpolate(kf2_depth_updated, size=(512, 512), mode="nearest")
    # save_depth_map(kf_interp.depths[0].detach().cpu().numpy(), save_root / 'kf2_depth.png', vmin=0, vmax=cutoff_depth*0.95, save_clean=True)
    # save_point_cloud_as_ply(kf_interp.additional_points_3d*500, kf_interp.run_dir / 'kf2_point_cloud.ply', kf_interp.additional_colors)
    # save_point_cloud_as_ply(kf_interp.points_3d *500, kf_interp.run_dir / 'kf1_point_cloud.ply', kf_interp.kf1_colors)
    evaluate_epoch(kf_interp, 0, vmax=vmax)

    for epoch in tqdm(range(1, total_frames + 1)):
        render_output_kf1 = kf_interp.render_kf1(epoch)

        inpaint_output = kf_interp.inpaint(render_output_kf1["rendered_image"], render_output_kf1["inpaint_mask"])

        if config["finetune_decoder_interp"]:
            finetune_decoder(config, kf_interp, render_output_kf1, inpaint_output, config["num_finetune_decoder_steps_interp"])

        # use latent to get fine-tuned image; center crop if needed; then update image/mask/depth
        kf_interp.update_images_and_masks(inpaint_output["latent"], render_output_kf1["inpaint_mask"])

        kf_interp.update_additional_point_cloud(render_output_kf1["rendered_depth"], kf_interp.images[-1], append_depth=True)

        # reload decoder
        kf_interp.vae.decoder = deepcopy(kf_interp.decoder_copy)
        with torch.no_grad():
            kf_interp.images_orig_decoder.append(kf_interp.decode_latents(inpaint_output["latent"]).detach())
        evaluate_epoch(kf_interp, epoch, vmax=cutoff_depth*0.95)
        empty_cache()

    kf_interp.images.append(kf1_image)  # so that the last frame is KF1
    evaluate(kf_interp)
    # save_point_cloud_as_ply(torch.cat([kf_interp.points_3d, kf_interp.additional_points_3d], dim=0)*500, kf_interp.run_dir / 'final_point_cloud.ply', torch.cat([kf_interp.kf1_colors, kf_interp.additional_colors], dim=0))

    # all_rundir.append(kf_interp.run_dir)
    return kf_interp.run_dir
