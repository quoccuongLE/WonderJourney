import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from omegaconf import OmegaConf
from PIL import Image
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor

sys.path.append('midas_module')

from models.inpainting_pipeline import get_inpainter
from models.keyframe_generate import keyframe_generate
from models.keyframe_interpolate import keyframe_interpolate
from util.chatGPT4 import TextpromptGen
from util.segment_utils import create_mask_generator
from util.utils import (load_example_yaml, merge_frames,
                        merge_keyframes, seeding)


def run(config):

    ###### ------------------ Load modules ------------------ ######

    if config['skip_gen']:
        kfgen_save_folder = Path(config['runs_dir']) / f"{config['kfgen_load_dt_string']}_kfgen"
    else:
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        kfgen_save_folder = Path(config['runs_dir']) / f"{dt_string}_kfgen"
    kfgen_save_folder.mkdir(exist_ok=True, parents=True)
    cutoff_depth = config['fg_depth_range'] + config['depth_shift']
    vmax = cutoff_depth * 2
    inpainting_resolution_gen = config['inpainting_resolution_gen']
    seeding(config["seed"])

    segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")

    mask_generator = create_mask_generator()

    all_rundir = []
    yaml_data = load_example_yaml(config["example_name"], 'examples/examples.yaml')
    start_keyframe = Image.open(yaml_data['image_filepath']).convert('RGB').resize((512, 512))
    content_prompt, style_prompt, adaptive_negative_prompt, background_prompt, control_text = yaml_data['content_prompt'], yaml_data['style_prompt'], yaml_data['negative_prompt'], yaml_data.get('background', None), yaml_data.get('control_text', None)
    if adaptive_negative_prompt != "":
        adaptive_negative_prompt += ", "
    all_keyframes = [start_keyframe]

    if isinstance(control_text, list):
        config['num_scenes'] = len(control_text)
    text_prompt_generator = TextpromptGen(config['runs_dir'], isinstance(control_text, list))
    content_list = content_prompt.split(',')
    scene_name = content_list[0]
    entities = content_list[1:]
    scene_dict = {'scene_name': scene_name, 'entities': entities, 'style': style_prompt, 'background': background_prompt}
    inpainting_prompt = style_prompt + ', ' + content_prompt

    inpainter_config = dict(
        model_name=config["stable_diffusion_checkpoint"], text_encoder="T5EncoderModel"
    )
    inpainter_pipeline, vae = get_inpainter(config=inpainter_config)
    rotation_path = config['rotation_path']
    assert len(rotation_path) >= config['num_scenes'] * config['num_keyframes']

    ###### ------------------ Main loop ------------------ ######

    for i in range(config['num_scenes']):
        if config['use_gpt']:
            control_text_this = control_text[i] if isinstance(control_text, list) else None
            scene_dict = text_prompt_generator.run_conversation(scene_name=scene_dict['scene_name'], entities=scene_dict['entities'], style=style_prompt, background=scene_dict['background'], control_text=control_text_this)
        inpainting_prompt = text_prompt_generator.generate_prompt(style=style_prompt, entities=scene_dict['entities'], background=scene_dict['background'], scene_name=scene_dict['scene_name'])

        for j in range(config['num_keyframes']):

            ###### ------------------ Keyframe (the major part of point clouds) generation ------------------ ######

            kf_gen_dict = keyframe_generate(
                config=config,
                kfgen_save_folder=kfgen_save_folder,
                scene_id=i,
                keyframe_id=j,
                rotation_path=rotation_path,
                inpainter_pipeline=inpainter_pipeline,
                inpainting_resolution_gen=inpainting_resolution_gen,
                vae=vae,
                mask_generator=mask_generator,
                segment_model=segment_model,
                segment_processor=segment_processor,
                vmax=vmax,
                cutoff_depth=cutoff_depth,
                text_prompt_generator=text_prompt_generator,
            )

            ###### ------------------ Keyframe interpolation (completing point clouds and rendering) ------------------ ######
            print("----------Keyframe interpolation----------")
            run_dir = keyframe_interpolate(
                scene_id=i,
                keyframe_id=j,
                config=config,
                kf_gen_dict=kf_gen_dict,
                inpainting_prompt=inpainting_prompt,
                inpainter_pipeline=inpainter_pipeline,
                vae=vae,
                vmax=vmax,
                cutoff_depth=cutoff_depth,
                rotation_path=rotation_path,
            )
            all_rundir.append(run_dir)

    dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
    save_dir = Path(config['runs_dir']) / f"{dt_string}_merged"
    if not config['skip_interp']:
        merge_frames(all_rundir, save_dir=save_dir, fps=config["save_fps"], is_forward=True, save_depth=False, save_gif=False)
    merge_keyframes(all_keyframes, save_dir=save_dir)
    text_prompt_generator.write_all_content(save_dir=save_dir)


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config"
    )
    args = parser.parse_args()
    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_config, example_config)

    POSTMORTEM = config['debug']
    if POSTMORTEM:
        try:
            run(config)
        except Exception as e:
            print(e)
            import ipdb
            ipdb.post_mortem()
    else:
        run(config)
