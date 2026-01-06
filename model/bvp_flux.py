"""
Flux BVP Solver for Geodesic Optimization

Adapts the boundary value problem solver for Flux architecture.
"""

import os
import torch
from typing import Optional, Dict, Tuple, List
from PIL import Image

from model.pipeline_flux import FluxGeodesicPipeline, load_flux_pipe
from model.score_flux import FluxScoreDistillation, FluxTextInversion
from model.utils import (
    lerp_cond_embed, o_project, o_project_, norm_fix, norm_fix_,
    display_alongside, display_in_two_rows
)
from model.spline import Spline
from model.bisection import Bisection_sampler


class FluxBVPOptimiser:
    """Learning rate scheduler for BVP optimization."""
    
    def __init__(
        self,
        iter_num: int,
        lr: float,
        lr_scheduler: str = 'constant',
        lr_divide: bool = True
    ):
        self.iter_num = iter_num
        self.lr_ini = lr
        self.lr_scheduler = lr_scheduler
        self.lr_divide = lr_divide

    def get_learning_rate(self, cur_iter: int, t: torch.Tensor) -> float:
        if self.lr_scheduler == 'constant':
            return self.lr_ini
        
        scale = cur_iter / self.iter_num
        
        if self.lr_scheduler == 'linear':
            cur_lr = self.lr_ini * (1 - scale)
        elif self.lr_scheduler == 'cosine':
            cur_lr = self.lr_ini * 0.5 * (1 + torch.cos(torch.tensor(torch.pi * scale)))
        elif self.lr_scheduler == 'polynomial':
            cur_lr = self.lr_ini * (1 - scale) ** 2
        else:
            raise ValueError(f'Unknown lr_scheduler: {self.lr_scheduler}')
        
        if self.lr_divide:
            cur_lr = cur_lr / len(t)
        
        return float(cur_lr)


class FluxBVPIO:
    """Input/Output handler for Flux BVP solver."""
    
    def __init__(
        self,
        pipe: FluxGeodesicPipeline,
        noise_level: float = 0.0,
        cfg_sample: float = 3.5,
        num_inference_steps: int = 8,
        out_dir: str = './',
        output_psample: bool = True,
        output_image_num: int = 17,
        output_start_images: bool = False,
        output_opt_points: bool = False,
        output_reconstruct_end: bool = False,
        output_separate_images: bool = True,
        out_interval: int = -1,
        use_lerp_cond: bool = False,
        imgA: Optional[Image.Image] = None,
        imgB: Optional[Image.Image] = None,
    ):
        self.pipe = pipe
        self.device = pipe.device
        self.noise_level = noise_level
        self.cfg_sample = cfg_sample
        self.num_inference_steps = num_inference_steps
        self.out_dir = out_dir
        self.output_psample = output_psample
        self.output_image_num = output_image_num
        self.output_start_images = output_start_images
        self.output_opt_points = output_opt_points
        self.output_reconstruct_end = output_reconstruct_end
        self.output_separate_images = output_separate_images
        self.out_interval = out_interval
        self.use_lerp_cond = use_lerp_cond
        self.imgA = imgA
        self.imgB = imgB
        
        self.out_t = torch.linspace(0, 1, self.output_image_num).to(self.device)
        
        # Create output directories
        os.makedirs(out_dir, exist_ok=True)
        if self.output_start_images:
            os.makedirs(os.path.join(self.out_dir, 'start_imgs'), exist_ok=True)
        os.makedirs(os.path.join(self.out_dir, 'out_imgs'), exist_ok=True)
        
        # Encode empty prompt for unconditional
        self.embed_uncond, self.pooled_uncond = pipe.prompt2embed('')
    
    def forward_single(
        self,
        input_data,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """Forward process: image/latent to noised latent."""
        if isinstance(input_data, torch.Tensor) and len(input_data.shape) == 4:
            lat = input_data
        else:
            lat = self.pipe.img2latent(input_data)
        
        if self.noise_level > 0:
            # For Flux, we use simple noise addition
            noise = torch.randn_like(lat)
            t = self.noise_level
            lat = (1 - t) * lat + t * noise
        
        return lat
    
    def backward_single(
        self,
        lat: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> Image.Image:
        """Backward process: noised latent to image."""
        if self.noise_level > 0:
            lat = self.pipe.latent_backward(
                lat,
                prompt_embeds,
                pooled_embeds,
                noise_level=self.noise_level,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.cfg_sample,
            )
        
        img = self.pipe.latent2img(lat)
        return img
    
    def backward_multi(
        self,
        X: torch.Tensor,
        prompt_embeds: torch.Tensor,
        pooled_embeds: torch.Tensor,
    ) -> List[Image.Image]:
        """Decode multiple latents to images."""
        imgs = []
        for i in range(X.shape[0]):
            lat = X[i:i+1].reshape(1, -1, 64, 64)  # Reshape from flat
            
            # Get corresponding embeddings
            if prompt_embeds.shape[0] > 1:
                pe = prompt_embeds[i:i+1]
                pp = pooled_embeds[i:i+1]
            else:
                pe = prompt_embeds
                pp = pooled_embeds
            
            img = self.backward_single(lat, pe, pp)
            imgs.append(img)
        
        return imgs


class FluxGradAnalysis:
    """Gradient analysis for debugging."""
    
    def __init__(self, grad_analysis_out: bool, grad_out_txt: str):
        self.grad_analysis_out = grad_analysis_out
        self.grad_out_txt = grad_out_txt
    
    def grad_analysis(
        self,
        t: torch.Tensor,
        cur_iter: int,
        grad_term1: torch.Tensor,
        grad_term2: torch.Tensor,
        grad_all: torch.Tensor,
    ) -> Tuple[float, float, float, float]:
        """Analyze gradient components."""
        grad_term1 = grad_term1.reshape(-1, grad_term1.shape[-1])
        grad_term2 = grad_term2.reshape(-1, grad_term2.shape[-1])
        grad_all = grad_all.reshape(-1, grad_all.shape[-1])
        
        # Compute angle between terms
        cos_theta = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        cos = cos_theta(grad_term1, grad_term2)
        angle = torch.arccos(cos.clamp(-1, 1)) * 180 / torch.pi
        
        # Compute norms
        g_norm1 = torch.norm(grad_term1, dim=-1)
        g_norm2 = torch.norm(grad_term2, dim=-1)
        g_norm_all = torch.norm(grad_all, dim=-1)
        
        g_n = torch.mean(g_norm_all).item()
        g1_n = torch.mean(g_norm1).item()
        g2_n = torch.mean(g_norm2).item()
        g_angle = torch.mean(angle).item()
        
        if self.grad_analysis_out and cur_iter % 10 == 0:
            with open(self.grad_out_txt, 'a') as f:
                f.write(f'iter:{cur_iter}\n')
                f.write(f't:{[round(ti.item(), 4) for ti in t]}\n')
                f.write(f'g_t1:{g1_n:.4f}, g_t2:{g2_n:.4f}, g_all:{g_n:.4f}, angle:{g_angle:.2f}\n\n')
        
        return g_n, g1_n, g2_n, g_angle


class Geodesic_BVP_Flux:
    """
    Geodesic Boundary Value Problem Solver using Flux.
    
    Computes geodesics in Flux latent space between two images.
    """
    
    def __init__(
        self,
        pipe: FluxGeodesicPipeline,
        imgA: Image.Image,
        imgB: Image.Image,
        promptA: str,
        promptB: str,
        noise_level: float,
        alpha: float,
        grad_args: Dict,
        bisect_args: Dict,
        output_args: Dict,
        tv_args: Dict,
        opt_args: Dict,
        spline_type: str = 'spherical_cubic',
        grad_analysis_out: bool = True,
        use_lerp_cond: bool = False,
        sphere_constraint: bool = True,
        test_name: str = 'test_bvp_flux',
        **kwargs
    ):
        self.pipe = pipe
        self.device = pipe.device
        self.imgA = imgA
        self.imgB = imgB
        self.promptA = promptA
        self.promptB = promptB
        self.test_name = test_name
        self.alpha = alpha
        self.sphere_constraint = sphere_constraint
        self.use_lerp_cond = use_lerp_cond
        
        # Convert noise_level to Flux timestep (0-1000)
        self.time_step = noise_level * 1000
        self.noise_level = noise_level
        
        # Initialize components
        self.score_dist = FluxScoreDistillation(
            pipe=pipe,
            time_step=self.time_step,
            **grad_args
        )
        
        self.optimizer = FluxBVPOptimiser(**opt_args)
        self.bisect_sampler = Bisection_sampler(**bisect_args)
        
        self.text_inv = FluxTextInversion(pipe=pipe, **tv_args)
        
        self.io = FluxBVPIO(
            pipe=pipe,
            noise_level=noise_level,
            imgA=imgA,
            imgB=imgB,
            use_lerp_cond=use_lerp_cond,
            **output_args
        )
        
        self.grad_analyzer = FluxGradAnalysis(
            grad_analysis_out,
            os.path.join(output_args.get('out_dir', './'), 'grad_check.txt')
        )
        
        # Optimization state
        self.cur_iter = 0
        self.path = {}
        
        # Initialize embeddings and path
        self._initialize()
    
    def _initialize(self):
        """Initialize embeddings and starting path."""
        print('[BVP-Flux] Initializing...')
        
        # Encode images to latents
        latA0 = self.pipe.img2latent(self.imgA)
        latB0 = self.pipe.img2latent(self.imgB)
        
        # Get latent dimensions
        self.latent_shape = latA0.shape  # [1, C, H, W]
        self.latent_dim = latA0.numel()  # Total elements
        
        print(f'[BVP-Flux] Latent shape: {self.latent_shape}, dim: {self.latent_dim}')
        
        # Text inversion for better conditioning
        if self.use_lerp_cond:
            self.embed_condA, self.pooled_condA = self.text_inv.text_inversion_load(
                self.promptA, latA0, self.test_name, 'A'
            )
            self.embed_condB, self.pooled_condB = self.text_inv.text_inversion_load(
                self.promptB, latB0, self.test_name, 'B'
            )
        else:
            prompt = f"{self.promptA} {self.promptB}"
            lat_combined = torch.cat([latA0, latB0], dim=0)
            self.embed_cond, self.pooled_cond = self.text_inv.text_inversion_load(
                prompt, lat_combined, self.test_name, 'AB'
            )
        
        # Forward process to get noised latents
        if self.use_lerp_cond:
            latA = self.io.forward_single(latA0, self.embed_condA, self.pooled_condA)
            latB = self.io.forward_single(latB0, self.embed_condB, self.pooled_condB)
        else:
            latA = self.io.forward_single(latA0, self.embed_cond, self.pooled_cond)
            latB = self.io.forward_single(latB0, self.embed_cond, self.pooled_cond)
        
        # Flatten to 1D for optimization
        pointA = latA.reshape(-1)
        pointB = latB.reshape(-1)
        
        # Apply sphere constraint
        if self.sphere_constraint:
            self.radius = 0.5 * (torch.norm(pointA) + torch.norm(pointB))
            pointA = norm_fix(pointA, self.radius)
            pointB = norm_fix(pointB, self.radius)
            print(f'[BVP-Flux] Sphere radius: {self.radius:.4f}')
        
        # Initialize spline
        self.spline = Spline('spherical_cubic')
        self.spline.fit_spline(
            torch.tensor([0.0, 1.0]).to(self.device),
            torch.stack([pointA, pointB], dim=0)
        )
        
        self.path[0] = pointA
        self.path[1] = pointB
        
        print('[BVP-Flux] Initialization complete')
    
    def _get_embeddings_at_t(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get interpolated embeddings at time t."""
        if self.use_lerp_cond:
            # Linear interpolation of embeddings
            prompt_embeds = torch.cat([
                (1 - ti) * self.embed_condA + ti * self.embed_condB
                for ti in t
            ], dim=0)
            pooled_embeds = torch.cat([
                (1 - ti) * self.pooled_condA + ti * self.pooled_condB
                for ti in t
            ], dim=0)
        else:
            prompt_embeds = self.embed_cond.repeat(len(t), 1, 1)
            pooled_embeds = self.pooled_cond.repeat(len(t), 1)
        
        return prompt_embeds, pooled_embeds
    
    def bvp_gradient(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        A: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], float, float]:
        """
        Compute BVP gradient for geodesic optimization.
        
        Args:
            X: Position on path [N, D]
            V: Velocity [N, D]
            A: Acceleration [N, D]
            t: Time parameters [N]
            
        Returns:
            grad: Gradient for optimization (or None if skipped)
            g_n: Gradient norm
            g_angle: Angle between gradient terms
        """
        # Reshape for score computation
        lats = X.reshape(-1, *self.latent_shape[1:])
        
        # Get embeddings
        prompt_embeds, pooled_embeds = self._get_embeddings_at_t(t)
        
        # Compute score distillation gradient
        d_logps = self.score_dist.grad_compute_batch(lats, prompt_embeds, pooled_embeds)
        d_logps = d_logps.reshape(-1, self.latent_dim)
        
        # Project if using sphere constraint
        if self.sphere_constraint:
            d_logps = o_project_(d_logps, X)
            A = o_project_(A, X)
        
        # Compute geodesic equation terms
        V_norm2 = torch.sum(V * V, dim=-1)
        A_scaled = A / V_norm2[:, None]
        
        term1 = o_project_(d_logps, V)
        term2 = o_project_(A_scaled, V) * (1 / self.alpha)
        
        grad = -(term1 + term2)
        
        # Analyze gradients
        g_n, g1_n, g2_n, g_angle = self.grad_analyzer.grad_analysis(
            t, self.cur_iter, term1, term2, grad
        )
        
        # Skip if acceleration term dominates (heuristic)
        if g1_n < g2_n:
            return None, g_n, g_angle
        
        return grad, g_n, g_angle
    
    def step(self) -> bool:
        """
        Perform one optimization step.
        
        Returns:
            finished: True if optimization is complete
        """
        # Get control points to optimize
        t_opt = self.bisect_sampler.get_control_t()
        if t_opt is None:
            return True
        
        t_opt = t_opt.to(self.device)
        
        # Get position, velocity, acceleration from spline
        X_opt = self.spline(t_opt)
        V_opt = self.spline(t_opt, 1)
        A_opt = self.spline(t_opt, 2)
        
        # Compute gradient
        grad, g_n, g_angle = self.bvp_gradient(X_opt, V_opt, A_opt, t_opt)
        
        if grad is None:
            # Skip this iteration, increase bisection strength
            self.bisect_sampler.add_strength(None)
            self.cur_iter += 1
            return False
        
        # Get learning rate
        cur_lr = self.optimizer.get_learning_rate(self.cur_iter, t_opt)
        
        # Log progress
        if self.cur_iter % 5 == 0:
            t_str = [f'{ti:.3f}' for ti in t_opt.cpu().numpy()]
            print(f'[BVP-Flux] iter={self.cur_iter}, t={t_str}, grad={g_n:.6f}, angle={g_angle:.1f}')
        
        # Update positions
        X_opt = X_opt - cur_lr * grad
        
        # Project back to sphere
        if self.sphere_constraint:
            X_opt = norm_fix_(X_opt, torch.tensor([self.radius] * X_opt.shape[0]).to(self.device))
        
        # Update path
        self.cur_iter += 1
        for i, ti in enumerate(t_opt):
            self.path[ti.item()] = X_opt[i]
        
        # Refit spline
        t_fit = torch.tensor(sorted(self.path.keys())).to(self.device)
        X_fit = torch.stack([self.path[ti.item()] for ti in t_fit], dim=0)
        self.spline.fit_spline(t_fit, X_fit)
        
        # Update bisection
        self.bisect_sampler.add_strength(self.cur_iter)
        
        return False
    
    def save_sequence(self, name: str):
        """Save current path as image sequence."""
        print(f'[BVP-Flux] Saving sequence: {name}')
        
        # Sample points along path
        t_out = self.io.out_t
        X_out = self.spline(t_out)
        
        # Get embeddings
        prompt_embeds, pooled_embeds = self._get_embeddings_at_t(t_out)
        
        # Decode to images
        if not self.io.output_reconstruct_end:
            # Use original images for endpoints
            imgs = [self.imgA]
            imgs.extend(self.io.backward_multi(X_out[1:-1], prompt_embeds[1:-1], pooled_embeds[1:-1]))
            imgs.append(self.imgB)
        else:
            imgs = self.io.backward_multi(X_out, prompt_embeds, pooled_embeds)
        
        # Save individual images
        if self.io.output_separate_images:
            save_dir = 'start_imgs' if name == 'start' else 'out_imgs'
            for i, img in enumerate(imgs):
                img.save(os.path.join(self.io.out_dir, save_dir, f'{i:02d}.png'))
        
        # Save concatenated image
        img_long = display_alongside(imgs)
        img_long.save(os.path.join(self.io.out_dir, f'long_{name}.png'))
        print(f'[BVP-Flux] Saved to {self.io.out_dir}/long_{name}.png')
        
        return imgs
    
    def solve(self):
        """Run the full BVP optimization."""
        print('[BVP-Flux] Starting optimization...')
        
        # Save initial path
        if self.io.output_start_images:
            self.save_sequence('start')
        
        # Optimization loop
        for i in range(self.optimizer.iter_num):
            finished = self.step()
            
            if finished or i == self.optimizer.iter_num - 1:
                self.save_sequence('final')
                break
            
            # Periodic output
            if self.io.out_interval > 0 and self.cur_iter % self.io.out_interval == 0:
                self.save_sequence(str(self.cur_iter))
            
            # Clear cache
            torch.cuda.empty_cache()
        
        # Save optimization points
        if self.io.output_opt_points:
            torch.save(self.path, os.path.join(self.io.out_dir, 'opt_points.pth'))
        
        print('[BVP-Flux] Optimization complete')