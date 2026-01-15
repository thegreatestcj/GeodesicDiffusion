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
        disable_text_encoder: bool = False,
    ):
        self.device = device
        self.dtype = dtype
        self.model_id = model_id
        self.disable_text_encoder = disable_text_encoder

        # Setup cache
        resolved_cache_dir = setup_cache_dir(cache_dir)
        print(f"[FluxPipeline] Model cache: {resolved_cache_dir}")
        print(f"[FluxPipeline] Loading {model_id}...")

        # Load pipeline - skip text encoders if disabled to save memory
        if disable_text_encoder:
            print("[FluxPipeline] Text encoders DISABLED - using null embeddings")
            self.pipe = FluxPipeline.from_pretrained(
                model_id,
                torch_dtype=dtype,
                cache_dir=os.path.join(resolved_cache_dir, 'hub'),
                text_encoder=None,
                text_encoder_2=None,
                tokenizer=None,
                tokenizer_2=None,
            )
        else:
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

        # Enable memory-efficient attention if available
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
            print("[FluxPipeline] xFormers memory-efficient attention enabled")
        except Exception:
            # xFormers not available - Flux uses PyTorch SDPA by default
            # Note: Do NOT use AttnProcessor2_0 with Flux - it's incompatible with FluxAttention
            print("[FluxPipeline] Using default Flux attention (PyTorch SDPA)")

        # Extract components
        self.transformer: FluxTransformer2DModel = self.pipe.transformer
        self.vae: AutoencoderKL = self.pipe.vae
        self.text_encoder: CLIPTextModel = self.pipe.text_encoder if not disable_text_encoder else None
        self.text_encoder_2: T5EncoderModel = self.pipe.text_encoder_2 if not disable_text_encoder else None
        self.tokenizer: CLIPTokenizer = self.pipe.tokenizer if not disable_text_encoder else None
        self.tokenizer_2: T5TokenizerFast = self.pipe.tokenizer_2 if not disable_text_encoder else None
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
        if self.text_encoder is not None:
            self.text_encoder.requires_grad_(False)
        if self.text_encoder_2 is not None:
            self.text_encoder_2.requires_grad_(False)

        # Cache null embeddings if text encoder is disabled
        self._null_prompt_embeds = None
        self._null_pooled_embeds = None
        
        self.generator = torch.Generator(device="cpu")
        print(f"[FluxPipeline] Loaded successfully")
    
    def set_seed(self, seed: int):
        self.generator = torch.Generator(device="cpu").manual_seed(seed)
    
    # =========================================================================
    # VAE Encoding/Decoding
    # =========================================================================
    
    def img2latent(
        self,
        image: Image.Image,
        height: int = None,
        width: int = None,
        max_size: int = None,
        min_size: int = None,
    ) -> torch.Tensor:
        """
        Encode image to latent space.

        Args:
            image: PIL Image to encode
            height: Target height (if None, uses image height rounded to nearest 16)
            width: Target width (if None, uses image width rounded to nearest 16)
            max_size: Maximum dimension (if None, no limit - adaptive to input)
            min_size: Minimum dimension (if None, no limit - adaptive to input)

        Returns:
            Latent tensor of shape [1, C, H//8, W//8]
        """
        # Get image dimensions
        img_width, img_height = image.size

        # Use provided dimensions or adapt from image
        if height is None:
            height = img_height
        if width is None:
            width = img_width

        # Round to nearest multiple of 16 (VAE requires multiple of 8, and Flux packing may require 16)
        height = ((height + 15) // 16) * 16
        width = ((width + 15) // 16) * 16

        # Apply max_size constraint if specified
        if max_size is not None and (height > max_size or width > max_size):
            # Scale down while preserving aspect ratio
            scale = min(max_size / height, max_size / width)
            height = int(height * scale)
            width = int(width * scale)
            # Re-round after scaling
            height = ((height + 15) // 16) * 16
            width = ((width + 15) // 16) * 16

        # Apply min_size constraint if specified
        if min_size is not None and (height < min_size or width < min_size):
            # Scale up while preserving aspect ratio
            scale = max(min_size / height, min_size / width)
            height = int(height * scale)
            width = int(width * scale)
            # Re-round after scaling
            height = ((height + 15) // 16) * 16
            width = ((width + 15) // 16) * 16

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
    
    def _get_null_embeddings(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get cached null embeddings for text-encoder-disabled mode.

        These are zeros with the correct shapes for Flux transformer.
        """
        if self._null_prompt_embeds is None:
            # Flux transformer expects:
            # - prompt_embeds: [batch, seq_len, hidden_size] - T5 output (4096 dim)
            # - pooled_embeds: [batch, pooled_dim] - CLIP pooled (768 dim)
            # Using minimal sequence length to save memory
            seq_len = 1  # Minimal sequence
            t5_hidden = 4096  # T5-XXL hidden size
            clip_pooled = 768  # CLIP pooled size

            self._null_prompt_embeds = torch.zeros(
                1, seq_len, t5_hidden,
                device=self.device, dtype=self.dtype
            )
            self._null_pooled_embeds = torch.zeros(
                1, clip_pooled,
                device=self.device, dtype=self.dtype
            )
            print(f"[FluxPipeline] Created null embeddings: prompt={self._null_prompt_embeds.shape}, pooled={self._null_pooled_embeds.shape}")

        return self._null_prompt_embeds, self._null_pooled_embeds

    def encode_prompt(
        self,
        prompt: str,
        prompt_2: Optional[str] = None,
        max_sequence_length: int = 512,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text prompt using dual encoders."""
        # Return null embeddings if text encoder is disabled
        if self.disable_text_encoder:
            return self._get_null_embeddings()

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
    disable_text_encoder: bool = False,
) -> FluxGeodesicPipeline:
    """Load Flux pipeline with recommended settings.

    Args:
        device: Torch device ('cuda' or 'cpu')
        enable_cpu_offload: Enable CPU offload for memory efficiency
        cache_dir: HuggingFace cache directory
        disable_text_encoder: If True, skip loading text encoders to save ~10GB VRAM
    """
    return FluxGeodesicPipeline(
        model_id="black-forest-labs/FLUX.1-dev",
        device=device,
        dtype=torch.bfloat16,
        enable_cpu_offload=enable_cpu_offload,
        cache_dir=cache_dir,
        disable_text_encoder=disable_text_encoder,
    )