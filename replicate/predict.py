import os
import sys
from typing import List
import torch
from pytorch_lightning import seed_everything
from cog import BasePredictor, Input, Path

sys.path.insert(0, "../src")
from japanese_stable_diffusion import JapaneseStableDiffusionPipeline


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print("Loading pipeline...")

        model_id = "rinna/japanese-stable-diffusion"
        cache_dir = "japanese-stable-diffusion-cache"
        # the model should be downloaded with auth token and saved to cache
        self.pipe = JapaneseStableDiffusionPipeline.from_pretrained(
            model_id,
            cache_dir=cache_dir,
            local_files_only=True,
        ).to("cuda")

    @torch.inference_mode()
    @torch.cuda.amp.autocast()
    def predict(
        self,
        prompt: str = Input(description="Input prompt", default="サラリーマン 油絵"),
        num_outputs: int = Input(
            description="Number of images to output", choices=[1, 4], default=1
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps", ge=1, le=500, default=50
        ),
        guidance_scale: float = Input(
            description="Scale for classifier-free guidance", ge=1, le=20, default=7.5
        ),
        seed: int = Input(
            description="Random seed. Leave blank to randomize the seed", default=None
        ),
    ) -> List[Path]:
        """Run a single prediction on the model"""

        if seed is None:
            seed = int.from_bytes(os.urandom(2), "big")
        print(f"Using seed: {seed}")

        seed_everything(seed)

        output = self.pipe(
            prompt=[prompt] * num_outputs,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
        )
        if any(output["nsfw_content_detected"]):
            raise Exception("NSFW content detected, please try a different prompt")

        output_paths = []
        for i, sample in enumerate(output["sample"]):
            output_path = f"/tmp/out-{i}.png"
            sample.save(output_path)
            output_paths.append(Path(output_path))

        return output_paths
