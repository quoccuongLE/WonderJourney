import json
from copy import deepcopy
from pathlib import Path
import torch
from torchvision.transforms import functional as F

from midas_module.midas.model_loader import load_model as load_depth_model
from models.models import KeyframeGen
from models.evaluate import evaluate_epoch
from util.utils import save_depth_map, seeding, empty_cache
from util.finetune_utils import finetune_depth_model, finetune_decoder


def keyframe_generate(
    config: dict,
    kfgen_save_folder: Path,
    scene_id: int,
    keyframe_id: int,
    rotation_path,
    inpainting_resolution_gen: int,
    inpainter_pipeline,
    mask_generator,
    vae,
    segment_model,
    segment_processor,
    vmax,
    cutoff_depth,
    text_prompt_generator,
    all_keyframes,
):
    if config['skip_gen']:
        kf_gen_dict = torch.load(kfgen_save_folder / f"s{scene_id:02d}_k{keyframe_id:01d}_gen_dict.pt")
        kf1_depth, kf2_depth = kf_gen_dict['kf1_depth'], kf_gen_dict['kf2_depth']
        kf1_image, kf2_image = kf_gen_dict['kf1_image'], kf_gen_dict['kf2_image']
        kf1_camera, kf2_camera = kf_gen_dict['kf1_camera'], kf_gen_dict['kf2_camera']
        kf2_mask = kf_gen_dict['kf2_mask']
        inpainting_prompt, adaptive_negative_prompt = kf_gen_dict['inpainting_prompt'], kf_gen_dict['adaptive_negative_prompt']
        rotation = kf_gen_dict['rotation']
    else:
        rotation = rotation_path[scene_id * config["num_keyframes"] + keyframe_id]
        regen_negative_prompt = ""
        config['inpainting_resolution_gen'] = inpainting_resolution_gen
        for regen_id in range(config['regenerate_times'] + 1):
            if regen_id > 0:
                seeding(-1)
            depth_model, _, _, _ = load_depth_model(
                device=torch.device("cuda:0"),
                model_path="weights/dpt_beit_large_512.pt",
                model_type="dpt_beit_large_512",
                optimize=False,
            )
            # depth_model = torch.hub.load("intel-isl/MiDaS", _dpt_depth_model_type)
            # first keyframe is loaded and estimated depth
            kf_gen = KeyframeGen(config, inpainter_pipeline, mask_generator, depth_model, vae, rotation, 
                                start_keyframe, inpainting_prompt, regen_negative_prompt + adaptive_negative_prompt,
                                segment_model=segment_model, segment_processor=segment_processor).to(config["device"])
            save_root = Path(kf_gen.run_dir) / "images"
            kf_idx = 0

            save_depth_map(kf_gen.depths[kf_idx].detach().cpu().numpy(), save_root / 'kf1_original', vmin=0, vmax=vmax)
            kf_gen.refine_disp_with_segments(kf_idx, background_depth_cutoff=cutoff_depth)
            save_depth_map(kf_gen.depths[kf_idx].detach().cpu().numpy(), save_root / 'kf1_processed', vmin=0, vmax=vmax)
            evaluate_epoch(kf_gen, kf_idx, vmax=vmax)

            kf_idx = 1
            render_output = kf_gen.render(kf_idx)
            inpaint_output = kf_gen.inpaint(render_output["rendered_image"], render_output["inpaint_mask"])

            regenerate_information = {}
            if config['enable_regenerate'] and regen_id <= config['regenerate_times'] -1:
                gpt_border, gpt_blur = text_prompt_generator.evaluate_image(F.to_pil_image(inpaint_output['inpainted_image'][0]), eval_blur=False)
                regenerate_information['gpt_border'] = gpt_border
                regenerate_information['gpt_blur'] = gpt_blur
                if gpt_border:
                    print("chatGPT-4 says the image has border!")
                    regen_negative_prompt += "border, "
                if gpt_blur:
                    print("chatGPT-4 says the image has blurry effect!")
                    regen_negative_prompt += "blur, "
                regenerate = gpt_border
            else:
                regenerate = False

            with open(save_root / 'regenerate_info.json', 'w') as json_file:
                json.dump(regenerate_information, json_file, indent=4)

            if not regenerate:
                break
            if regen_id == config['regenerate_times'] -1:
                print("Regenerating faild after {} times".format(config['regenerate_times']))
                if gpt_border:
                    print("Use crop to solve border problem!")
                    config['inpainting_resolution_gen'] = 560
                else:
                    break

            # get memory back
            depth_model = kf_gen.depth_model.to('cpu')
            kf_gen.depth_model = None
            del depth_model
            empty_cache()

        if config["finetune_decoder_gen"]:
            F.to_pil_image(inpaint_output["inpainted_image"].detach()[0]).save(save_root / 'kf2_before_ft.png')
            finetune_decoder(config, kf_gen, render_output, inpaint_output, config['num_finetune_decoder_steps'])

        kf_gen.update_images_and_masks(inpaint_output["latent"], render_output["inpaint_mask"])

        kf2_depth_should_be = render_output['rendered_depth']
        mask_to_align_depth = ~(render_output["inpaint_mask_512"]>0) & (kf2_depth_should_be < cutoff_depth + kf_gen.kf_delta_t)
        mask_to_cutoff_depth = ~(render_output["inpaint_mask_512"]>0) & (kf2_depth_should_be >= cutoff_depth + kf_gen.kf_delta_t)

        # with torch.no_grad():
        #     kf2_before_ft_depth, _ = kf_gen.get_depth(kf_gen.images[kf_idx])  # pix depth under kf2 frame
        if config["finetune_depth_model"]:
            finetune_depth_model(config, kf_gen, kf2_depth_should_be, kf_idx, mask_align=mask_to_align_depth, 
                                mask_cutoff=mask_to_cutoff_depth, cutoff_depth=cutoff_depth + kf_gen.kf_delta_t)
        with torch.no_grad():
            kf2_ft_depth_original, kf2_ft_disp_original = kf_gen.get_depth(kf_gen.images[kf_idx])
            kf_gen.depths.append(kf2_ft_depth_original), kf_gen.disparities.append(kf2_ft_disp_original)
        # save_depth_map(kf2_before_ft_depth.detach().cpu().numpy(), save_root / 'kf2_before_ft_depth', vmin=0, vmax=vmax)
        # save_depth_map(kf2_depth_should_be_processed.detach().cpu().numpy(), save_root / 'kf2_depth_should_be_processed', vmin=0, vmax=vmax)
        # save_depth_map(kf2_depth_should_be_original.detach().cpu().numpy(), save_root / 'kf2_depth_should_be_original', vmin=0, vmax=vmax)
        # save_depth_map(kf2_ft_depth_original.cpu().numpy(), save_root / 'kf2_ft_depth_original', vmin=0, vmax=vmax)

        # get memory back
        depth_model = kf_gen.depth_model.to('cpu')
        kf_gen.depth_model = None
        del depth_model
        empty_cache()

        kf_gen.refine_disp_with_segments(kf_idx, background_depth_cutoff=cutoff_depth + kf_gen.kf_delta_t)
        save_depth_map(kf_gen.depths[-1].cpu().numpy(), save_root / 'kf2_ft_depth_processed', vmin=0, vmax=vmax)

        kf_gen.vae.decoder = deepcopy(kf_gen.decoder_copy)
        evaluate_epoch(kf_gen, kf_idx, vmax=vmax)

        start_keyframe = F.to_pil_image(kf_gen.images[1][0])
        all_keyframes.append(start_keyframe)

        kf1_depth, kf2_depth = kf_gen.depths[0], kf_gen.depths[-1]
        kf1_image, kf2_image = kf_gen.images[0], kf_gen.images[1]
        kf1_camera, kf2_camera = kf_gen.cameras[0], kf_gen.cameras[1]
        kf2_mask = render_output["inpaint_mask_512"]
        kf_gen_dict = {'kf1_depth': kf1_depth, 'kf2_depth': kf2_depth, 'kf1_image': kf1_image, 'kf2_image': kf2_image, 
                    'kf1_camera': kf1_camera, 'kf2_camera': kf2_camera, 'kf2_mask': kf2_mask, 'inpainting_prompt': inpainting_prompt, 
                    'adaptive_negative_prompt': adaptive_negative_prompt, 'rotation': rotation}
        torch.save(
            kf_gen_dict,
            kfgen_save_folder / f"s{scene_id:02d}_k{keyframe_id:01d}_gen_dict.pt",
        )
        if config['skip_interp']:
            kf_gen = kf_gen.to('cpu')
            del kf_gen
            empty_cache()
    return kf_gen_dict
