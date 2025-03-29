import torch
from diffusers import (AutoencoderKL, DPMSolverMultistepScheduler,
                       StableDiffusionInpaintPipeline)
from transformers import T5EncoderModel, BitsAndBytesConfig

from models.inpainting_pipeline.pipeline_pixart_inpaint import \
    PixArtAlphaInpaintPipeline
from util.utils import prepare_scheduler


def get_inpainter(config: dict[str, any], device: str = "cuda:0"):
    if config["model_name"] == "PixArt-alpha/PixArt-XL-2-512x512":
        if config.get("text_encoder"):
            text_encoder = T5EncoderModel.from_pretrained(
                "PixArt-alpha/PixArt-XL-2-1024-MS",
                subfolder="text_encoder",
                quantization_config=BitsAndBytesConfig(load_in_8bit=True),
                device_map="auto"
            )
        else:
            text_encoder = None
        inpainter_pipeline = PixArtAlphaInpaintPipeline.from_pretrained(
            pretrained_model_name_or_path=config["model_name"],
            text_encoder=text_encoder,
            use_safetensors=True,
            torch_dtype=torch.float16,
            # transformer=None,
        ).to(device)
        vae = inpainter_pipeline.vae
    else:
        inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            config["model_name"],
            safety_checker=None,
            torch_dtype=torch.float16,
            revision="fp16",
        ).to(device)
        inpainter_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(inpainter_pipeline.scheduler.config)
        inpainter_pipeline.scheduler = prepare_scheduler(inpainter_pipeline.scheduler)
        vae = AutoencoderKL.from_pretrained(config["model_name"], subfolder="vae").to(device)
    return inpainter_pipeline, vae
