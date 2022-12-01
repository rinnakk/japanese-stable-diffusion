import os
import torch
from torch import autocast
from diffusers import LMSDiscreteScheduler
from japanese_stable_diffusion import JapaneseStableDiffusionPipeline
from dotenv import load_dotenv


base_dir = os.path.dirname(os.path.dirname(__file__))
load_dotenv()
ACCESS_TOKEN = os.getenv("ACCESS_TOKEN")


def test_inference_jsd():
    model_id = "rinna/japanese-stable-diffusion"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use the K-LMS scheduler here instead
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000
    )
    pipe = JapaneseStableDiffusionPipeline.from_pretrained(
        pretrained_model_name_or_path=model_id,
        scheduler=scheduler,
        use_auth_token=ACCESS_TOKEN
    ).to(device)
    prompt = "猫の肖像画"
    image = pipe(prompt, guidance_scale=7.5, num_inference_steps=1)["images"][0]

    # image.save("output.png")
