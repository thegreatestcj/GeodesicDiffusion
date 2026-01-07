"""
Flux Score Distillation for Geodesic Computation

Adapts the score distillation approach for Flux's Flow Matching framework.

Key differences from SD:
1. Flux predicts velocity v = ε - x_0, not noise ε
2. Score estimation: ∇log p ≈ -v / σ(t)
3. Different timestep handling (continuous vs discrete)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Union

from model.pipeline_flux import FluxGeodesicPipeline


class FluxScoreDistillation:
    """
    Score distillation for Flux pipeline.
    
    Computes ∇log p(x) using the pretrained Flux model,
    which is needed for geodesic optimization.
    """
    
    def __init__(
        self,
        pipe: FluxGeodesicPipeline,
        time_step: float = 500,  # Flux uses timesteps in [0, 1000]
        grad_sample_range: float = 100,
        grad_weight_type: str = 'uniform',
        grad_guidance_0: float = 1.0,  # Weight for conditional score
        grad_guidance_1: float = 1.0,  # Weight for negative prompt
        grad_sample_type: str = 'back_n_forward_sample',
        grad_batch_size: int = 4,  # Smaller batch for Flux (larger model)
    ):
        """
        Args:
            pipe: FluxGeodesicPipeline instance
            time_step: Base timestep for score estimation
            grad_sample_range: Range for timestep sampling
            grad_weight_type: Weighting scheme ('uniform', 'increase', 'decrease')
            grad_guidance_0: Weight for positive prompt direction
            grad_guidance_1: Weight for negative prompt direction  
            grad_sample_type: Sampling strategy for timesteps
            grad_batch_size: Batch size for gradient computation
        """
        self.pipe = pipe
        self.device = pipe.device
        self.dtype = pipe.dtype
        
        self.time_step = time_step
        self.grad_sample_range = grad_sample_range
        self.grad_weight_type = grad_weight_type
        self.grad_guidance_0 = grad_guidance_0
        self.grad_guidance_1 = grad_guidance_1
        self.grad_sample_type = grad_sample_type
        self.grad_batch_size = grad_batch_size
        
        # Encode empty prompt for unconditional score
        self.embed_uncond, self.pooled_uncond = pipe.prompt2embed('')
        
        # Encode negative prompt for OOD handling
        neg_prompt = (
            "A doubling image, unrealistic, artifacts, distortions, "
            "unnatural blending, ghosting effects, overlapping edges, "
            "harsh transitions, motion blur, poor resolution, low detail"
        )
        self.embed_neg, self.pooled_neg = pipe.prompt2embed(neg_prompt)
    
    def get_sigma(self, t: torch.Tensor) -> torch.Tensor:
        """
        Get noise level σ(t) for Flow Matching.
        
        For linear interpolation: σ(t) = t / 1000 (normalized)
        """
        return (t / 1000).clamp(min=1e-5)
    
    def grad_weight(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient weight based on timestep.
        """
        if self.grad_weight_type == 'uniform':
            return torch.ones_like(t)
        
        sigma = self.get_sigma(t)
        
        if self.grad_weight_type == 'increase':
            # Higher weight at higher noise levels
            return sigma
        elif self.grad_weight_type == 'decrease':
            # Higher weight at lower noise levels
            return 1.0 / sigma.clamp(min=0.1)
        else:
            return torch.ones_like(t)
    
    def grad_prepare(
        self,
        latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare latent and timestep for gradient computation.
        
        Args:
            latent: Input latent [B, C, H, W]
            
        Returns:
            latent_t: Noised latent at sampled timestep
            t: Sampled timestep
        """
        batch_size = latent.shape[0]
        
        if self.grad_sample_type == 'ori_step':
            # Use fixed timestep
            t = torch.tensor([self.time_step], device=self.device).expand(batch_size)
            return latent, t
        
        elif self.grad_sample_type == 'forward_sample':
            # Sample from [time_step, time_step + range]
            if self.grad_sample_range == -1:
                max_t = 1000
            else:
                max_t = self.time_step + self.grad_sample_range
            
            t = torch.randint(
                int(self.time_step), int(max_t),
                (batch_size,), device=self.device
            ).float()
        
        elif self.grad_sample_type == 'back_n_forward_sample':
            # Sample from [time_step - range, time_step + range]
            if self.grad_sample_range == -1:
                range_t = min(self.time_step - 50, 950 - self.time_step)
            else:
                range_t = self.grad_sample_range
            
            min_t = max(50, self.time_step - range_t)
            max_t = min(950, self.time_step + range_t)
            
            t = torch.randint(
                int(min_t), int(max_t),
                (batch_size,), device=self.device
            ).float()
        
        else:
            raise ValueError(f'Unknown grad_sample_type: {self.grad_sample_type}')
        
        # Add noise to latent using Flow Matching
        noise = torch.randn_like(latent)
        sigma = self.get_sigma(t).view(-1, 1, 1, 1)
        
        # x_t = (1 - σ) * x_0 + σ * noise
        latent_t = (1 - sigma) * latent + sigma * noise
        
        return latent_t, t
    
    def grad_compute(
        self,
        latent: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute score distillation gradient.
        
        Uses Noise-Free Score Distillation (NFSD) approach adapted for Flux.
        
        Args:
            latent: Input latent [B, C, H, W]
            prompt_embeds: Conditional text embeddings
            pooled_embeds: Pooled text embeddings
            
        Returns:
            grad: Score gradient [B, C, H, W]
        """
        batch_size = latent.shape[0]
        
        # Expand embeddings for batch
        if prompt_embeds.shape[0] == 1:
            prompt_embeds = prompt_embeds.expand(batch_size, -1, -1)
            pooled_embeds = pooled_embeds.expand(batch_size, -1)
        
        embed_uncond = self.embed_uncond.expand(batch_size, -1, -1)
        pooled_uncond = self.pooled_uncond.expand(batch_size, -1)
        embed_neg = self.embed_neg.expand(batch_size, -1, -1)
        pooled_neg = self.pooled_neg.expand(batch_size, -1)
        
        # Prepare noised latent and timestep
        latent_t, t = self.grad_prepare(latent)
        latent_t = latent_t.to(dtype=self.dtype)
        
        grad_c = 0
        grad_d = 0
        
        with torch.no_grad():  # No gradients needed for score computation
            with torch.autocast(device_type='cuda', dtype=self.dtype):
                if self.grad_guidance_0 == self.grad_guidance_1:
                    # Optimization: single forward pass difference
                    # v_cond - v_neg gives direction toward conditional distribution
                    v_cond = self.pipe.predict_velocity(
                        latent_t, t, prompt_embeds, pooled_embeds, guidance_scale=1.0
                    )
                    grad_c = -v_cond.clone()
                    del v_cond
                    torch.cuda.empty_cache()

                    v_neg = self.pipe.predict_velocity(
                        latent_t, t, embed_neg, pooled_neg, guidance_scale=1.0
                    )
                    grad_d = v_neg.clone()
                    del v_neg
                    torch.cuda.empty_cache()
                else:
                    # Full computation with unconditional baseline
                    v_uncond = self.pipe.predict_velocity(
                        latent_t, t, embed_uncond, pooled_uncond, guidance_scale=1.0
                    )

                    if self.grad_guidance_0 > 0:
                        v_cond = self.pipe.predict_velocity(
                            latent_t, t, prompt_embeds, pooled_embeds, guidance_scale=1.0
                        )
                        grad_c = v_uncond - v_cond  # Direction toward conditional
                        del v_cond
                        torch.cuda.empty_cache()

                    if self.grad_guidance_1 > 0:
                        v_neg = self.pipe.predict_velocity(
                            latent_t, t, embed_neg, pooled_neg, guidance_scale=1.0
                        )
                        grad_d = v_neg - v_uncond  # Direction away from negative
                        del v_neg
                        torch.cuda.empty_cache()

                    del v_uncond
                    torch.cuda.empty_cache()
        
        # Weight and combine
        w = self.grad_weight(t).view(-1, 1, 1, 1)
        normalise = 1.0 / (abs(self.grad_guidance_0) + abs(self.grad_guidance_1))
        
        grad = w * normalise * (self.grad_guidance_0 * grad_c + self.grad_guidance_1 * grad_d)
        
        return grad.float()  # Return in float32 for stability
    
    def grad_compute_batch(
        self,
        latents: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradients in batches to manage memory.
        
        Args:
            latents: Input latents [N, C, H, W]
            prompt_embeds: Conditional embeddings [N, seq, dim]
            pooled_embeds: Pooled embeddings [N, dim]
            
        Returns:
            grad: Score gradients [N, C, H, W]
        """
        n = latents.shape[0]
        grad_out = []
        
        j = 0
        while j < n:
            batch_end = min(j + self.grad_batch_size, n)
            
            lats = latents[j:batch_end]
            embeds = prompt_embeds[j:batch_end] if prompt_embeds.shape[0] > 1 else prompt_embeds
            pooled = pooled_embeds[j:batch_end] if pooled_embeds.shape[0] > 1 else pooled_embeds
            
            grad = self.grad_compute(lats, embeds, pooled)
            grad_out.append(grad)
            
            j = batch_end
            
            # Clear cache after each batch
            torch.cuda.empty_cache()
        
        return torch.cat(grad_out, dim=0)


class FluxTextInversion:
    """
    Text inversion for Flux pipeline.
    
    Optimizes text embeddings to better represent specific images.
    """
    
    def __init__(
        self,
        pipe: FluxGeodesicPipeline,
        tv_lr: float = 0.005,
        tv_steps: int = 500,
        tv_batch_size: int = 2,
        tv_ckpt_folder: str = 'tv_ckpt_flux/',
    ):
        self.pipe = pipe
        self.tv_lr = tv_lr
        self.tv_steps = tv_steps
        self.tv_batch_size = tv_batch_size
        self.tv_ckpt_folder = tv_ckpt_folder
        
        import os
        os.makedirs(tv_ckpt_folder, exist_ok=True)
    
    def text_inversion(
        self,
        prompt: str,
        target_latent: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimize text embeddings to match target latent.
        
        Args:
            prompt: Initial text prompt
            target_latent: Target latent to match [B, C, H, W]
            
        Returns:
            prompt_embeds: Optimized prompt embeddings
            pooled_embeds: Optimized pooled embeddings
        """
        from tqdm import tqdm
        import torch.nn.functional as F
        
        # Get initial embeddings
        prompt_embeds, pooled_embeds = self.pipe.prompt2embed(prompt)

        # Skip optimization if tv_steps is 0 (memory-saving mode)
        if self.tv_steps == 0:
            print('[TextInv] Skipping optimization (tv_steps=0), using original embeddings')
            return prompt_embeds, pooled_embeds

        print('[TextInv] Optimizing text embeddings...')
        
        # Make copies for optimization
        prompt_embeds_opt = prompt_embeds.clone().requires_grad_(True)
        pooled_embeds_opt = pooled_embeds.clone().requires_grad_(True)
        
        optimizer = torch.optim.AdamW(
            [prompt_embeds_opt, pooled_embeds_opt],
            lr=self.tv_lr
        )
        
        target_latent = target_latent.to(dtype=self.pipe.dtype)
        if target_latent.shape[0] < self.tv_batch_size:
            target_latent = target_latent.repeat(self.tv_batch_size, 1, 1, 1)
        
        # Enable gradient checkpointing to reduce memory usage
        if hasattr(self.pipe.transformer, 'enable_gradient_checkpointing'):
            self.pipe.transformer.enable_gradient_checkpointing()

        for i in tqdm(range(self.tv_steps), desc='TextInv'):
            optimizer.zero_grad()

            # Sample random timesteps
            t = torch.randint(100, 900, (self.tv_batch_size,), device=self.pipe.device).float()

            # Add noise
            noise = torch.randn_like(target_latent[:self.tv_batch_size])
            sigma = (t / 1000).view(-1, 1, 1, 1)
            noisy_latent = (1 - sigma) * target_latent[:self.tv_batch_size] + sigma * noise

            # Predict velocity
            v_pred = self.pipe.predict_velocity(
                noisy_latent,
                t,
                prompt_embeds_opt.expand(self.tv_batch_size, -1, -1),
                pooled_embeds_opt.expand(self.tv_batch_size, -1),
                guidance_scale=1.0,
            )

            # Target velocity
            v_target = noise - target_latent[:self.tv_batch_size]

            # Loss
            loss = F.mse_loss(v_pred, v_target)
            loss.backward()
            optimizer.step()

            # Clear cache periodically to prevent fragmentation
            if i % 10 == 0:
                torch.cuda.empty_cache()

            if i % 100 == 0:
                print(f'[TextInv] Step {i}, loss: {loss.item():.6f}')
        
        prompt_embeds_opt = prompt_embeds_opt.detach()
        pooled_embeds_opt = pooled_embeds_opt.detach()
        
        return prompt_embeds_opt, pooled_embeds_opt
    
    def text_inversion_load(
        self,
        prompt: str,
        target_latent: torch.Tensor,
        prefix: str,
        postfix: str = '',
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Load or compute text inversion embeddings.
        """
        import os
        
        ckpt_name = f'{prefix}_{self.tv_steps}_{str(self.tv_lr).replace(".", "")}_{postfix}.pt'
        ckpt_path = os.path.join(self.tv_ckpt_folder, ckpt_name)
        
        print(f'[TextInv] Checkpoint path: {ckpt_path}')
        
        if os.path.exists(ckpt_path):
            print(f'[TextInv] Loading from cache...')
            data = torch.load(ckpt_path, weights_only=True)
            prompt_embeds = data['prompt_embeds'].to(self.pipe.device)
            pooled_embeds = data['pooled_embeds'].to(self.pipe.device)
        else:
            prompt_embeds, pooled_embeds = self.text_inversion(prompt, target_latent)
            torch.save({
                'prompt_embeds': prompt_embeds.cpu(),
                'pooled_embeds': pooled_embeds.cpu(),
            }, ckpt_path)
        
        return prompt_embeds, pooled_embeds