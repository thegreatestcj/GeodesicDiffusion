"""
Flux Pipeline for Geodesic Computation

Flux uses:
- DiT (Diffusion Transformer) instead of UNet
- Flow Matching instead of DDPM/DDIM
- Dual text encoders (CLIP + T5)
- Different latent space format

Reference: https://github.com/black-forest-labs/flux
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, List, Union
from PIL import Image
import numpy as np
from tqdm import tqdm

from diffusers import FluxPipeline, FluxTransformer2DModel
from diffusers.models import AutoencoderKL
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast

from model.utils import load_image, output_image, load_image_batch, output_image_batch


class FluxGeodesicPipeline:
    """
    Flux pipeline adapted for geodesic computation.
    
    Key differences from SD:
    1. Uses Flow Matching: x_t = (1-t)*x_0 + t*noise, velocity v = noise - x_0
    2. DiT architecture instead of UNet
    3. Different VAE (same architecture, different weights)
    4. Dual text encoders: CLIP for local features, T5 for global understanding
    """
    
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = True,
        enable_xformers: bool = True,
    ):
        """
        Initialize Flux pipeline.
        
        Args:
            model_id: HuggingFace model ID
            device: Target device
            dtype: Model precision (bf16 recommended for Flux)
            enable_cpu_offload: Enable sequential CPU offload to save VRAM
        """
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        
        print(f"[FluxPipeline] Loading {model_id}...")
        print(f"[FluxPipeline] This may take a few minutes for first download (~30GB)")
        
        # Load the full pipeline
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
        )

        # Enable xformers if requested
        if enable_xformers:
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
                print("[FluxPipeline] xformers memory efficient attention enabled")
            except Exception as e:
                print(f"[FluxPipeline] xformers not available: {e}")
                print("[FluxPipeline] Falling back to default attention")
        
        if enable_cpu_offload:
            # Sequential CPU offload: moves modules to GPU only when needed
            # Essential for 24GB VRAM
            self.pipe.enable_sequential_cpu_offload()
            print("[FluxPipeline] CPU offload enabled")
        else:
            self.pipe.to(device)
        
        # Extract components for direct access
        self.transformer: FluxTransformer2DModel = self.pipe.transformer
        self.vae: AutoencoderKL = self.pipe.vae
        self.text_encoder: CLIPTextModel = self.pipe.text_encoder
        self.text_encoder_2: T5EncoderModel = self.pipe.text_encoder_2
        self.tokenizer: CLIPTokenizer = self.pipe.tokenizer
        self.tokenizer_2: T5TokenizerFast = self.pipe.tokenizer_2
        self.scheduler = self.pipe.scheduler
        
        # Flux latent space config
        self.vae_scale_factor = self.pipe.vae_scale_factor  # Usually 8
        self.latent_channels = self.transformer.config.in_channels  # Usually 16 for Flux
        
        # Disable gradients for inference
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        
        # Random generator
        self.generator = torch.Generator(device="cpu")
        
        print(f"[FluxPipeline] Loaded successfully")
        print(f"[FluxPipeline] Latent channels: {self.latent_channels}")
        print(f"[FluxPipeline] VAE scale factor: {self.vae_scale_factor}")
    
    def set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # =========================================================================
    # VAE Encoding/Decoding
    # =========================================================================
    
    def img2latent(self, image: Image.Image, height: int = 512, width: int = 512) -> torch.Tensor:
        """
        Encode image to latent space.
        
        Args:
            image: PIL Image
            height: Target height
            width: Target width
            
        Returns:
            latent: Tensor of shape [1, C, H//8, W//8]
        """
        # Preprocess image
        img_tensor = load_image(image, self.device, resize_dims=(height, width))
        img_tensor = img_tensor.to(dtype=self.dtype)
        
        # Encode
        with torch.no_grad():
            latent_dist = self.vae.encode(img_tensor).latent_dist
            latent = latent_dist.sample() * self.vae.config.scaling_factor
        
        return latent
    
    def latent2img(self, latent: torch.Tensor) -> Image.Image:
        """
        Decode latent to image.
        
        Args:
            latent: Tensor of shape [1, C, H, W]
            
        Returns:
            image: PIL Image
        """
        latent = latent.to(dtype=self.dtype)
        
        with torch.no_grad():
            latent_scaled = latent / self.vae.config.scaling_factor
            img_tensor = self.vae.decode(latent_scaled).sample
        
        # Convert to PIL
        img = output_image(img_tensor.float())
        return img
    
    def img2latent_batch(self, images: List[Image.Image], height: int = 512, width: int = 512) -> torch.Tensor:
        """Encode multiple images to latents."""
        latents = []
        for img in images:
            latent = self.img2latent(img, height, width)
            latents.append(latent)
        return torch.cat(latents, dim=0)
    
    def latent2img_batch(self, latents: torch.Tensor) -> List[Image.Image]:
        """Decode multiple latents to images."""
        images = []
        for i in range(latents.shape[0]):
            img = self.latent2img(latents[i:i+1])
            images.append(img)
        return images
    
    # =========================================================================
    # Text Encoding
    # =========================================================================
    
    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode text prompt using dual encoders.
        
        Args:
            prompt: Text prompt
            prompt_2: Optional second prompt for T5 (defaults to prompt)
            max_sequence_length: Max tokens for T5
            
        Returns:
            prompt_embeds: CLIP embeddings [1, seq_len, hidden_dim]
            pooled_prompt_embeds: Pooled CLIP embeddings [1, hidden_dim]
        """
        prompt_2 = prompt_2 or prompt
        
        # Use the pipeline's encoding method
        (
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids
        ) = self.pipe.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            device=self.device,
            max_sequence_length=max_sequence_length,
        )
        
        return prompt_embeds, pooled_prompt_embeds
    
    def prompt2embed(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Simplified prompt encoding (compatible with SD interface).
        
        Returns tuple of (prompt_embeds, pooled_prompt_embeds)
        """
        return self.encode_prompt(prompt)
    
    # =========================================================================
    # Flow Matching Forward/Backward
    # =========================================================================
    
    def get_timestep(self, noise_level: float) -> torch.Tensor:
        """
        Convert noise level [0, 1] to Flux timestep.
        
        In Flow Matching: t=0 is clean, t=1 is noise
        """
        # Flux uses continuous timesteps in [0, 1]
        t = torch.tensor([noise_level], device=self.device, dtype=self.dtype)
        return t
    
    def add_noise(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Add noise using Flow Matching interpolation.
        
        x_t = (1 - t) * x_0 + t * noise
        
        Args:
            latent: Clean latent x_0
            noise: Noise tensor
            timestep: Time in [0, 1]
            
        Returns:
            noisy_latent: x_t
        """
        t = timestep.view(-1, 1, 1, 1)
        noisy_latent = (1 - t) * latent + t * noise
        return noisy_latent
    
    def get_velocity(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute ground truth velocity for Flow Matching.
        
        v = noise - x_0 (derivative of x_t w.r.t. t)
        """
        return noise - latent
    
    def latent_forward(
        self,
        latent: torch.Tensor,
        noise_level: float = 1.0,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward process: add noise to latent.
        
        Args:
            latent: Clean latent
            noise_level: Amount of noise [0, 1]
            noise: Optional pre-generated noise
            
        Returns:
            noisy_latent: Noised latent
        """
        if noise_level == 0:
            return latent
        
        if noise is None:
            noise = torch.randn_like(latent)
        
        t = self.get_timestep(noise_level)
        noisy_latent = self.add_noise(latent, noise, t)
        
        return noisy_latent
    
    def latent_backward(
        self,
        latent: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        noise_level: float = 1.0,
        num_inference_steps: int = 8,
        guidance_scale: float = 3.5,
    ) -> torch.Tensor:
        """
        Backward process: denoise latent using Flow Matching.
        
        Args:
            latent: Noisy latent at noise_level
            prompt_embeds: Text embeddings
            pooled_prompt_embeds: Pooled text embeddings
            noise_level: Starting noise level
            num_inference_steps: Number of denoising steps
            guidance_scale: CFG scale
            
        Returns:
            clean_latent: Denoised latent
        """
        # Set up timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=self.device)
        
        # Find starting index based on noise_level
        timesteps = self.scheduler.timesteps
        start_idx = int((1 - noise_level) * len(timesteps))
        timesteps = timesteps[start_idx:]
        
        # Prepare latent
        latent = latent.to(dtype=self.dtype)
        
        # Get text IDs (required by Flux)
        batch_size = latent.shape[0]
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3, device=self.device)
        
        # Get latent image IDs
        latent_image_ids = self._get_latent_image_ids(latent)
        
        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Denoising")):
            # Expand timestep for batch
            timestep = t.expand(batch_size)
            
            # Predict velocity
            velocity_pred = self._predict_velocity(
                latent,
                timestep,
                prompt_embeds,
                pooled_prompt_embeds,
                text_ids,
                latent_image_ids,
                guidance_scale,
            )
            
            # Step
            latent = self.scheduler.step(
                velocity_pred, t, latent, return_dict=False
            )[0]
        
        return latent
    
    def _predict_velocity(
        self,
        latent: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        text_ids: torch.Tensor,
        latent_image_ids: torch.Tensor,
        guidance_scale: float = 3.5,
    ) -> torch.Tensor:
        """
        Predict velocity using the transformer.
        
        Uses classifier-free guidance if guidance_scale > 1.
        """
        batch_size = latent.shape[0]
        
        # Pack latents to Flux format
        latent_packed = self._pack_latents(latent)
        
        # Expand embeddings for batch
        if prompt_embeds.shape[0] != batch_size:
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.expand(batch_size, -1)
        
        # Forward through transformer
        with torch.no_grad():
            velocity_pred = self.transformer(
                hidden_states=latent_packed,
                timestep=timestep / 1000,  # Flux expects normalized timestep
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=text_ids,
                img_ids=latent_image_ids,
                guidance=torch.tensor([guidance_scale], device=self.device, dtype=self.dtype),
                return_dict=False,
            )[0]
        
        # Unpack
        velocity_pred = self._unpack_latents(velocity_pred, latent.shape)
        
        return velocity_pred
    
    def _pack_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Pack latents to Flux transformer format.
        
        Flux expects: [B, (H*W), C] where patches are flattened
        """
        batch_size, channels, height, width = latents.shape
        
        # Reshape to sequence format
        latents = latents.view(batch_size, channels, height * width)
        latents = latents.permute(0, 2, 1)  # [B, H*W, C]
        
        return latents
    
    def _unpack_latents(self, latents: torch.Tensor, target_shape: Tuple) -> torch.Tensor:
        """
        Unpack latents from Flux transformer format.
        """
        batch_size, seq_len, channels = latents.shape
        height = width = int(seq_len ** 0.5)
        
        latents = latents.permute(0, 2, 1)  # [B, C, H*W]
        latents = latents.view(batch_size, channels, height, width)
        
        return latents
    
    def _get_latent_image_ids(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Generate positional IDs for latent patches.
        """
        batch_size, channels, height, width = latents.shape
        
        # Create grid of positions
        latent_image_ids = torch.zeros(height, width, 3, device=self.device, dtype=self.dtype)
        latent_image_ids[..., 1] = torch.arange(height, device=self.device)[:, None]
        latent_image_ids[..., 2] = torch.arange(width, device=self.device)[None, :]
        
        # Flatten and expand for batch
        latent_image_ids = latent_image_ids.view(1, height * width, 3)
        latent_image_ids = latent_image_ids.expand(batch_size, -1, -1)
        
        return latent_image_ids
    
    # =========================================================================
    # Score/Velocity Prediction for Geodesics
    # =========================================================================
    
    def predict_velocity(
        self,
        latent: torch.Tensor,
        timestep: Union[float, torch.Tensor],
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        guidance_scale: float = 1.0,
    ) -> torch.Tensor:
        """
        Predict velocity at given timestep (for geodesic computation).
        
        Args:
            latent: Latent tensor [B, C, H, W]
            timestep: Scalar or tensor timestep in [0, 1000]
            prompt_embeds: Text embeddings
            pooled_prompt_embeds: Pooled embeddings
            guidance_scale: CFG scale
            
        Returns:
            velocity: Predicted velocity [B, C, H, W]
        """
        batch_size = latent.shape[0]
        
        # Convert timestep
        if isinstance(timestep, float):
            timestep = torch.tensor([timestep], device=self.device)
        timestep = timestep.expand(batch_size)
        
        # Get IDs
        text_ids = torch.zeros(batch_size, prompt_embeds.shape[1], 3, device=self.device)
        latent_image_ids = self._get_latent_image_ids(latent)
        
        # Predict
        velocity = self._predict_velocity(
            latent,
            timestep,
            prompt_embeds,
            pooled_prompt_embeds,
            text_ids,
            latent_image_ids,
            guidance_scale,
        )
        
        return velocity
    
    def velocity_to_score(
        self,
        velocity: torch.Tensor,
        timestep: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """
        Convert velocity prediction to score (gradient of log probability).
        
        In Flow Matching:
        - x_t = (1-t)*x_0 + t*ε
        - v = dx_t/dt = ε - x_0
        - score ≈ -v / σ(t) where σ(t) = t (for linear interpolation)
        
        Args:
            velocity: Predicted velocity
            timestep: Current timestep
            
        Returns:
            score: Approximated score function
        """
        if isinstance(timestep, float):
            timestep = torch.tensor([timestep], device=self.device)
        
        # For Flow Matching with linear interpolation
        # σ(t) = t, so score ≈ -velocity / t
        sigma = timestep.view(-1, 1, 1, 1).clamp(min=1e-5)
        score = -velocity / sigma
        
        return score


def load_flux_pipe(device: str = 'cuda', enable_cpu_offload: bool = True, enable_xformers: bool = True) -> FluxGeodesicPipeline:
    """
    Load Flux pipeline with recommended settings.
    
    Args:
        device: Target device
        enable_cpu_offload: Enable CPU offload for VRAM saving
        
    Returns:
        pipe: FluxGeodesicPipeline instance
    """
    pipe = FluxGeodesicPipeline(
        model_id="black-forest-labs/FLUX.1-dev",
        device=device,
        dtype=torch.bfloat16,
        enable_cpu_offload=enable_cpu_offload,
        enable_xformers=enable_xformers,
    )
    return pipe