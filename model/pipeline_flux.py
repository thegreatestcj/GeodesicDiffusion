"""
Flux Pipeline for Geodesic Computation

IMPORTANT: This version dynamically adapts to the transformer's expected input format
by checking transformer.config.in_channels at runtime.

Different diffusers versions handle Flux packing differently:
- Some expect raw channels (16) with internal packing
- Some expect pre-packed channels (64) with 2x2 patches

This implementation handles both cases automatically.
"""

import os
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


def setup_cache_dir(cache_dir: Optional[str] = None) -> str:
    """Setup HuggingFace cache directory for cluster environments."""
    if cache_dir is not None:
        resolved_dir = cache_dir
    elif os.environ.get('HF_HOME'):
        resolved_dir = os.environ['HF_HOME']
    elif os.environ.get('SCRATCH'):
        resolved_dir = os.path.join(os.environ['SCRATCH'], 'huggingface')
    else:
        resolved_dir = os.path.join(os.path.expanduser('~'), '.cache', 'huggingface')
    
    os.makedirs(resolved_dir, exist_ok=True)
    os.environ['HF_HOME'] = resolved_dir
    os.environ['HUGGINGFACE_HUB_CACHE'] = os.path.join(resolved_dir, 'hub')
    
    return resolved_dir


class FluxGeodesicPipeline:
    """
    Flux pipeline adapted for geodesic computation.
    
    Automatically detects the transformer's expected input format and
    adapts packing/unpacking accordingly.
    """
    
    def __init__(
        self,
        model_id: str = "black-forest-labs/FLUX.1-dev",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        enable_cpu_offload: bool = True,
        cache_dir: Optional[str] = None,
    ):
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        
        # Setup cache
        resolved_cache_dir = setup_cache_dir(cache_dir)
        print(f"[FluxPipeline] Model cache: {resolved_cache_dir}")
        print(f"[FluxPipeline] Loading {model_id}...")
        
        # Load pipeline
        self.pipe = FluxPipeline.from_pretrained(
            model_id,
            torch_dtype=dtype,
            cache_dir=os.path.join(resolved_cache_dir, 'hub'),
        )
        
        if enable_cpu_offload:
            # Use model_cpu_offload instead of sequential_cpu_offload
            # sequential_cpu_offload can hang on some systems
            self.pipe.enable_model_cpu_offload()
            print("[FluxPipeline] CPU offload enabled (model-level)")
        else:
            self.pipe.to(device)

        # Extract components
        self.transformer: FluxTransformer2DModel = self.pipe.transformer
        self.vae: AutoencoderKL = self.pipe.vae
        self.text_encoder: CLIPTextModel = self.pipe.text_encoder
        self.text_encoder_2: T5EncoderModel = self.pipe.text_encoder_2
        self.tokenizer: CLIPTokenizer = self.pipe.tokenizer
        self.tokenizer_2: T5TokenizerFast = self.pipe.tokenizer_2
        self.scheduler = self.pipe.scheduler

        # NOTE: Gradient checkpointing is incompatible with sequential CPU offload
        # Enable it only when needed for training (in score_flux.py)

        # Get VAE latent channels
        self.vae_scale_factor = self.pipe.vae_scale_factor  # typically 8
        self.vae_latent_channels = self.vae.config.latent_channels  # typically 16

        # CRITICAL: Detect transformer's expected input format
        self.transformer_in_channels = self.transformer.config.in_channels

        # Determine packing mode
        if self.transformer_in_channels == self.vae_latent_channels:
            # No packing - transformer handles it internally
            self.packing_mode = 'none'
            self.patch_size = 1
        elif self.transformer_in_channels == self.vae_latent_channels * 4:
            # 2x2 patch packing required
            self.packing_mode = '2x2'
            self.patch_size = 2
        else:
            raise ValueError(
                f"Unexpected configuration: transformer expects {self.transformer_in_channels} channels, "
                f"but VAE outputs {self.vae_latent_channels} channels"
            )

        print(f"[FluxPipeline] VAE latent channels: {self.vae_latent_channels}")
        print(f"[FluxPipeline] Transformer in_channels: {self.transformer_in_channels}")
        print(f"[FluxPipeline] Packing mode: {self.packing_mode}")

        # Disable gradients for inference
        self.transformer.requires_grad_(False)
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        self.text_encoder_2.requires_grad_(False)
        
        self.generator = torch.Generator(device="cpu")
        print(f"[FluxPipeline] Loaded successfully")
    
    def set_seed(self, seed: int):
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # =========================================================================
    # VAE Encoding/Decoding
    # =========================================================================
    
    def img2latent(self, image: Image.Image, height: int = 512, width: int = 512) -> torch.Tensor:
        """Encode image to latent space."""
        img_tensor = load_image(image, self.device, resize_dims=(height, width))
        img_tensor = img_tensor.to(dtype=self.dtype)
        
        with torch.no_grad():
            latent_dist = self.vae.encode(img_tensor).latent_dist
            latent = latent_dist.sample() * self.vae.config.scaling_factor
        
        return latent
    
    def latent2img(self, latent: torch.Tensor) -> Image.Image:
        """Decode latent to image."""
        latent = latent.to(dtype=self.dtype)
        
        with torch.no_grad():
            latent_scaled = latent / self.vae.config.scaling_factor
            img_tensor = self.vae.decode(latent_scaled).sample
        
        img = output_image(img_tensor.float())
        return img
    
    # =========================================================================
    # Text Encoding
    # =========================================================================
    
    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt using dual encoders."""
        prompt_2 = prompt_2 or prompt
        
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
        """Simplified prompt encoding."""
        return self.encode_prompt(prompt)
    
    # =========================================================================
    # Dynamic Latent Packing/Unpacking
    # =========================================================================
    
    def _pack_latents(self, latents: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Pack latents for transformer based on detected packing mode.
        
        Returns:
            packed_latents: Packed tensor ready for transformer
            info: Dictionary with unpacking information (height, width, etc.)
        """
        batch_size, channels, height, width = latents.shape
        
        info = {
            'batch_size': batch_size,
            'channels': channels,
            'height': height,
            'width': width,
        }
        
        if self.packing_mode == 'none':
            # Simple flatten: [B, C, H, W] -> [B, H*W, C]
            packed = latents.view(batch_size, channels, height * width)
            packed = packed.permute(0, 2, 1)  # [B, H*W, C]
            info['seq_len'] = height * width
            
        elif self.packing_mode == '2x2':
            # 2x2 patch packing: [B, C, H, W] -> [B, (H/2)*(W/2), C*4]
            packed = latents.view(
                batch_size, channels, 
                height // 2, 2, 
                width // 2, 2
            )
            packed = packed.permute(0, 2, 4, 1, 3, 5)  # [B, H/2, W/2, C, 2, 2]
            packed = packed.reshape(
                batch_size, 
                (height // 2) * (width // 2), 
                channels * 4
            )
            info['seq_len'] = (height // 2) * (width // 2)
        
        return packed, info
    
    def _unpack_latents(self, latents: torch.Tensor, info: dict) -> torch.Tensor:
        """
        Unpack latents from transformer output back to image format.
        """
        batch_size = info['batch_size']
        channels = info['channels']
        height = info['height']
        width = info['width']
        
        if self.packing_mode == 'none':
            # [B, H*W, C] -> [B, C, H, W]
            unpacked = latents.permute(0, 2, 1)  # [B, C, H*W]
            unpacked = unpacked.view(batch_size, channels, height, width)
            
        elif self.packing_mode == '2x2':
            # [B, (H/2)*(W/2), C*4] -> [B, C, H, W]
            unpacked = latents.reshape(
                batch_size,
                height // 2, width // 2,
                channels, 2, 2
            )
            unpacked = unpacked.permute(0, 3, 1, 4, 2, 5)  # [B, C, H/2, 2, W/2, 2]
            unpacked = unpacked.reshape(batch_size, channels, height, width)
        
        return unpacked
    
    def _get_latent_image_ids(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Generate positional IDs based on packing mode.
        """
        batch_size, channels, height, width = latents.shape
        
        if self.packing_mode == 'none':
            # Position IDs for each pixel
            seq_h, seq_w = height, width
        else:
            # Position IDs for each 2x2 patch
            seq_h, seq_w = height // 2, width // 2
        
        latent_image_ids = torch.zeros(
            seq_h, seq_w, 3,
            device=latents.device,
            dtype=latents.dtype
        )
        
        latent_image_ids[..., 1] = torch.arange(
            seq_h, device=latents.device, dtype=latents.dtype
        )[:, None]
        latent_image_ids[..., 2] = torch.arange(
            seq_w, device=latents.device, dtype=latents.dtype
        )[None, :]
        
        latent_image_ids = latent_image_ids.view(1, seq_h * seq_w, 3)
        latent_image_ids = latent_image_ids.expand(batch_size, -1, -1)
        
        return latent_image_ids
    
    # =========================================================================
    # Flow Matching Operations
    # =========================================================================
    
    def get_timestep(self, noise_level: float) -> torch.Tensor:
        """Convert noise level [0, 1] to timestep tensor."""
        return torch.tensor([noise_level], device=self.device, dtype=self.dtype)
    
    def add_noise(
        self,
        latent: torch.Tensor,
        noise: torch.Tensor,
        timestep: torch.Tensor
    ) -> torch.Tensor:
        """Flow Matching noise addition: x_t = (1-t)*x_0 + t*noise"""
        t = timestep.view(-1, 1, 1, 1)
        return (1 - t) * latent + t * noise
    
    def latent_forward(
        self,
        latent: torch.Tensor,
        noise_level: float = 1.0,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward process: add noise to latent."""
        if noise_level == 0:
            return latent
        if noise is None:
            noise = torch.randn_like(latent)
        t = self.get_timestep(noise_level)
        return self.add_noise(latent, noise, t)
    
    def latent_backward(
        self,
        latent: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_prompt_embeds: torch.Tensor,
        noise_level: float = 1.0,
        num_inference_steps: int = 8,
        guidance_scale: float = 3.5,
    ) -> torch.Tensor:
        """Backward process: denoise latent."""
        batch_size = latent.shape[0]

        # Get image sequence length for dynamic shifting (required by FlowMatchEulerDiscreteScheduler)
        _, channels, height, width = latent.shape
        if self.packing_mode == '2x2':
            seq_len = (height // 2) * (width // 2)
        else:
            seq_len = height * width

        # Compute mu for dynamic shifting if the scheduler requires it
        scheduler_kwargs = {"device": self.device}
        if getattr(self.scheduler.config, 'use_dynamic_shifting', False):
            # Flux uses image_seq_len to compute mu internally
            scheduler_kwargs["mu"] = self.pipe.scheduler.config.base_shift + (
                self.pipe.scheduler.config.max_shift - self.pipe.scheduler.config.base_shift
            ) * (seq_len / (seq_len + 1))

        self.scheduler.set_timesteps(num_inference_steps, **scheduler_kwargs)
        timesteps = self.scheduler.timesteps
        
        start_idx = int((1 - noise_level) * len(timesteps))
        timesteps = timesteps[start_idx:]
        
        latent = latent.to(dtype=self.dtype)
        
        # Get position IDs
        text_ids = torch.zeros(
            batch_size, prompt_embeds.shape[1], 3, 
            device=self.device, dtype=self.dtype
        )
        latent_image_ids = self._get_latent_image_ids(latent)
        
        for t in tqdm(timesteps, desc="Denoising"):
            timestep = t.expand(batch_size)
            
            velocity_pred = self._predict_velocity_internal(
                latent, timestep, prompt_embeds, pooled_prompt_embeds,
                text_ids, latent_image_ids, guidance_scale,
            )
            
            latent = self.scheduler.step(velocity_pred, t, latent, return_dict=False)[0]
        
        return latent
    
    def _predict_velocity_internal(
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
        Internal velocity prediction with pre-computed IDs.
        """
        batch_size = latent.shape[0]

        # Pack latents (keeping original dtype to preserve gradients if needed)
        latent_packed, pack_info = self._pack_latents(latent)

        # Expand embeddings for batch
        if prompt_embeds.shape[0] != batch_size:
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
            pooled_prompt_embeds = pooled_prompt_embeds.expand(batch_size, -1)

        # Use autocast for mixed precision - this preserves gradients while handling dtype
        with torch.autocast(device_type='cuda', dtype=self.dtype):
            velocity_pred = self.transformer(
                hidden_states=latent_packed,
                timestep=timestep / 1000,  # Normalize to [0, 1]
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                txt_ids=text_ids.squeeze(0) if text_ids.dim() == 3 else text_ids,
                img_ids=latent_image_ids.squeeze(0) if latent_image_ids.dim() == 3 else latent_image_ids,
                guidance=torch.tensor([guidance_scale], device=self.device, dtype=self.dtype),
                return_dict=False,
            )[0]

        # Unpack back to image format
        velocity_pred = self._unpack_latents(velocity_pred, pack_info)

        return velocity_pred
    
    # =========================================================================
    # Public Velocity Prediction API (for geodesic computation)
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
        Predict velocity at given timestep.
        
        This is the main API for geodesic computation.
        """
        batch_size = latent.shape[0]
        
        if isinstance(timestep, float):
            timestep = torch.tensor([timestep], device=self.device, dtype=self.dtype)
        timestep = timestep.expand(batch_size).to(dtype=self.dtype)
        
        # Compute IDs
        text_ids = torch.zeros(
            batch_size, prompt_embeds.shape[1], 3,
            device=self.device, dtype=self.dtype
        )
        latent_image_ids = self._get_latent_image_ids(latent)
        
        velocity = self._predict_velocity_internal(
            latent, timestep, prompt_embeds, pooled_prompt_embeds,
            text_ids, latent_image_ids, guidance_scale,
        )
        
        return velocity
    
    def velocity_to_score(
        self,
        velocity: torch.Tensor,
        timestep: Union[float, torch.Tensor],
    ) -> torch.Tensor:
        """Convert velocity to score: score ≈ -v/σ(t)"""
        if isinstance(timestep, float):
            timestep = torch.tensor([timestep], device=self.device)
        sigma = timestep.view(-1, 1, 1, 1).clamp(min=1e-5)
        return -velocity / sigma


def load_flux_pipe(
    device: str = 'cuda',
    enable_cpu_offload: bool = True,
    cache_dir: Optional[str] = None,
) -> FluxGeodesicPipeline:
    """Load Flux pipeline with recommended settings."""
    return FluxGeodesicPipeline(
        model_id="black-forest-labs/FLUX.1-dev",
        device=device,
        dtype=torch.bfloat16,
        enable_cpu_offload=enable_cpu_offload,
        cache_dir=cache_dir,
    )