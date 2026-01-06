"""
Flux BVP Solver for Geodesic Optimization with Semantic Regularization

Adapts the boundary value problem solver for Flux architecture.
Includes semantic regularization using DINO-v2 features for improved
semantic consistency along the interpolation path.

Based on: "Probability Density Geodesics in Image Diffusion Latent Space"
(Yu et al., CVPR 2025)
"""

import os
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
from PIL import Image
from torchvision import transforms

from model.pipeline_flux import FluxGeodesicPipeline, load_flux_pipe
from model.score_flux import FluxScoreDistillation, FluxTextInversion
from model.utils import (
    lerp_cond_embed, o_project, o_project_, norm_fix, norm_fix_,
    display_alongside, display_in_two_rows
)
from model.spline import Spline
from model.bisection import Bisection_sampler


# =============================================================================
# Semantic Regularizer for Flux
# =============================================================================

class FluxSemanticRegularizer:
    """
    Computes semantic consistency regularization using DINO-v2 features.
    
    Encourages the interpolation path to maintain semantic coherence between
    adjacent frames by penalizing large jumps in DINO feature space.
    
    Adapted for Flux's latent space format.
    """
    
    def __init__(self, device='cuda', model_name='dinov2_vits14'):
        """
        Args:
            device: Torch device ('cuda' or 'cpu')
            model_name: DINO model variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14')
        """
        self.device = device
        self.model_name = model_name
        
        print(f'[FluxSemanticReg] Loading {model_name}...')
        self.dino = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dino.eval()
        self.dino.to(device)
        
        # Freeze DINO parameters
        for param in self.dino.parameters():
            param.requires_grad = False
        
        # DINO preprocessing: resize to 224x224, normalize with ImageNet stats
        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f'[FluxSemanticReg] {model_name} loaded successfully.')
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract DINO-v2 CLS token features from images.
        
        Args:
            images: Tensor of shape [B, 3, H, W] in range [0, 1]
            
        Returns:
            features: Tensor of shape [B, feature_dim]
        """
        # Clamp to valid range
        images = torch.clamp(images, 0, 1)
        images_preprocessed = self.preprocess(images)
        
        with torch.no_grad():
            features = self.dino(images_preprocessed)
        
        return features
    
    def compute_gradient(
        self,
        latents: torch.Tensor,
        pipe: FluxGeodesicPipeline,
        V: torch.Tensor,
        latent_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Compute semantic regularization gradient for Flux latents.
        
        Args:
            latents: Tensor of shape [N, D] - flattened latent codes
            pipe: FluxGeodesicPipeline for VAE decoding
            V: Tensor of shape [N, D] - velocity vectors
            latent_shape: Original latent shape (C, H, W)
            
        Returns:
            grad: Tensor of shape [N, D] - semantic regularization gradient
        """
        N = latents.shape[0]
        latent_dim = latents.shape[1]
        
        if N < 2:
            return torch.zeros_like(latents)
        
        # Reshape latents to 4D for VAE decoding
        latents_4d = latents.reshape(N, *latent_shape)
        
        # Decode latents to images
        with torch.no_grad():
            # Scale latents for VAE decoding
            scaled_latents = latents_4d / pipe.vae.config.scaling_factor
            scaled_latents = scaled_latents.to(dtype=pipe.dtype)
            
            # Decode in batches to save memory
            images_list = []
            batch_size = 2  # Smaller batch for Flux (larger VAE)
            for i in range(0, N, batch_size):
                batch = scaled_latents[i:i+batch_size]
                decoded = pipe.vae.decode(batch)['sample']
                images_list.append(decoded.float())
            
            images = torch.cat(images_list, dim=0)  # [N, 3, H, W]
            images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
            # Extract DINO features
            features = self.extract_features(images)  # [N, feat_dim]
        
        # Compute feature differences between adjacent frames
        feat_diff = features[1:] - features[:-1]  # [N-1, feat_dim]
        
        # Build gradient: push each point toward its neighbors in feature space
        grad_features = torch.zeros_like(features)
        grad_features[1:] += feat_diff      # Contribution from left neighbor
        grad_features[:-1] -= feat_diff     # Contribution from right neighbor
        
        # Project to latent space using random projection
        grad_latents = self._project_to_latent(grad_features, latent_dim, latents.device)
        
        return grad_latents
    
    def _project_to_latent(
        self,
        grad_features: torch.Tensor,
        latent_dim: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Project feature-space gradient to latent space using random projection.
        
        Args:
            grad_features: Tensor of shape [N, feat_dim]
            latent_dim: Target latent dimension
            device: Torch device
            
        Returns:
            grad_latents: Tensor of shape [N, latent_dim]
        """
        N, feat_dim = grad_features.shape
        
        # Normalize for stability
        grad_norm = torch.norm(grad_features, dim=1, keepdim=True) + 1e-8
        grad_normalized = grad_features / grad_norm
        
        # Fixed random projection (seeded for reproducibility)
        generator = torch.Generator(device=device).manual_seed(42)
        projection = torch.randn(feat_dim, latent_dim, device=device, generator=generator)
        projection = F.normalize(projection, dim=1)
        
        # Project and restore magnitude
        grad_latents = torch.mm(grad_normalized, projection)
        grad_latents = grad_latents * grad_norm
        
        return grad_latents


# =============================================================================
# BVP Optimizer Base Class
# =============================================================================

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


# =============================================================================
# BVP Input/Output Handler
# =============================================================================

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
        latent_shape: Tuple[int, ...]
    ) -> List[Image.Image]:
        """Decode multiple latents to images."""
        imgs = []
        for i in range(X.shape[0]):
            lat = X[i:i+1].reshape(1, *latent_shape)  # Reshape from flat
            
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


# =============================================================================
# Gradient Analysis
# =============================================================================

class FluxGradAnalysis:
    """Gradient analysis for debugging and logging."""
    
    def __init__(self, grad_analysis_out: bool, grad_out_txt: str):
        self.grad_analysis_out = grad_analysis_out
        self.grad_out_txt = grad_out_txt
    
    def grad_analysis(
        self,
        t: torch.Tensor,
        cur_iter: int,
        grad_term1: torch.Tensor,
        grad_term2: torch.Tensor,
        grad_term3: torch.Tensor,
        grad_all: torch.Tensor,
    ) -> Tuple[float, float, float, float, float]:
        """
        Analyze gradient components including semantic term.
        
        Returns:
            g_n: Total gradient norm
            g1_n: Term1 (score distillation) norm
            g2_n: Term2 (smoothness) norm
            g3_n: Term3 (semantic) norm
            g_angle: Angle between term1 and term2
        """
        grad_term1 = grad_term1.reshape(-1, grad_term1.shape[-1])
        grad_term2 = grad_term2.reshape(-1, grad_term2.shape[-1])
        grad_term3 = grad_term3.reshape(-1, grad_term3.shape[-1])
        grad_all = grad_all.reshape(-1, grad_all.shape[-1])
        
        # Compute angle between term1 and term2
        cos_theta = torch.nn.CosineSimilarity(dim=1, eps=1e-8)
        cos = cos_theta(grad_term1, grad_term2)
        angle = torch.arccos(cos.clamp(-1, 1)) * 180 / torch.pi
        
        # Compute norms
        g_norm1 = torch.norm(grad_term1, dim=-1)
        g_norm2 = torch.norm(grad_term2, dim=-1)
        g_norm3 = torch.norm(grad_term3, dim=-1)
        g_norm_all = torch.norm(grad_all, dim=-1)
        
        g_n = torch.mean(g_norm_all).item()
        g1_n = torch.mean(g_norm1).item()
        g2_n = torch.mean(g_norm2).item()
        g3_n = torch.mean(g_norm3).item()
        g_angle = torch.mean(angle).item()
        
        if self.grad_analysis_out and cur_iter % 10 == 0:
            with open(self.grad_out_txt, 'a') as f:
                f.write(f'iter:{cur_iter}\n')
                f.write(f't:{[round(ti.item(), 4) for ti in t]}\n')
                f.write(f'g_t1(score):{g1_n:.6f}, g_t2(smooth):{g2_n:.6f}, '
                       f'g_t3(semantic):{g3_n:.6f}, g_all:{g_n:.6f}, angle:{g_angle:.2f}\n\n')
        
        return g_n, g1_n, g2_n, g3_n, g_angle


# =============================================================================
# Main Geodesic BVP Solver for Flux
# =============================================================================

class Geodesic_BVP_Flux:
    """
    Geodesic Boundary Value Problem Solver using Flux with Semantic Regularization.
    
    Computes geodesics in Flux latent space between two images.
    
    The gradient consists of three terms:
        term1: Probability density gradient (via score distillation)
        term2: Smoothness constraint (acceleration penalty from geodesic equation)
        term3: Semantic consistency (DINO feature continuity)
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
        semantic_args: Optional[Dict] = None,
        spline_type: str = 'spherical_cubic',
        grad_analysis_out: bool = True,
        use_lerp_cond: bool = False,
        sphere_constraint: bool = True,
        test_name: str = 'test_bvp_flux',
        **kwargs
    ):
        """
        Initialize the Flux BVP solver with semantic regularization.
        
        Args:
            pipe: FluxGeodesicPipeline instance
            imgA, imgB: Input images (endpoints)
            promptA, promptB: Text prompts for conditioning
            noise_level: Diffusion noise level [0, 1]
            alpha: Geodesic smoothness weight (beta in paper)
            grad_args: Score distillation arguments
            bisect_args: Bisection sampler arguments
            output_args: Output/IO arguments
            tv_args: Text inversion arguments
            opt_args: Optimizer arguments
            semantic_args: Semantic regularization arguments (NEW)
                - use_semantic_reg: bool, enable semantic regularization
                - semantic_weight: float, weight for semantic term
                - dino_model: str, DINO model variant
            spline_type: Spline interpolation type
            grad_analysis_out: Enable gradient analysis logging
            use_lerp_cond: Use linear embedding interpolation
            sphere_constraint: Constrain path to sphere
            test_name: Test identifier for checkpoints
        """
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
        
        # Initialize score distillation
        self.score_dist = FluxScoreDistillation(
            pipe=pipe,
            time_step=self.time_step,
            **grad_args
        )
        
        # Initialize optimizer
        self.optimizer = FluxBVPOptimiser(**opt_args)
        
        # Initialize bisection sampler
        self.bisect_sampler = Bisection_sampler(**bisect_args)
        
        # Initialize text inversion
        self.text_inv = FluxTextInversion(pipe=pipe, **tv_args)
        
        # Initialize IO handler
        self.io = FluxBVPIO(
            pipe=pipe,
            noise_level=noise_level,
            imgA=imgA,
            imgB=imgB,
            use_lerp_cond=use_lerp_cond,
            **output_args
        )
        
        # Initialize gradient analyzer (will be updated after semantic init)
        self.grad_analyzer = FluxGradAnalysis(
            grad_analysis_out,
            os.path.join(output_args.get('out_dir', './'), 'grad_check.txt')
        )
        
        # Optimization state
        self.cur_iter = 0
        self.path = {}
        
        # =====================================================================
        # Initialize semantic regularization
        # =====================================================================
        if semantic_args is None:
            semantic_args = {}
        
        # Support both semantic_args dict and kwargs for backward compatibility
        self.use_semantic_reg = semantic_args.get(
            'use_semantic_reg', kwargs.get('use_semantic_reg', False)
        )
        self.semantic_weight = semantic_args.get(
            'semantic_weight', kwargs.get('semantic_weight', 0.1)
        )
        dino_model = semantic_args.get('dino_model', 'dinov2_vits14')
        
        if self.use_semantic_reg:
            print(f'[BVP-Flux] Semantic regularization enabled, weight={self.semantic_weight}')
            self.semantic_reg = FluxSemanticRegularizer(
                device=str(self.device), model_name=dino_model
            )
        else:
            print(f'[BVP-Flux] Semantic regularization disabled')
            self.semantic_reg = None
        
        # Initialize embeddings and path
        self._initialize()
    
    def _initialize(self):
        """Initialize embeddings and starting path."""
        print('[BVP-Flux] Initializing...')
        
        # Encode images to latents
        latA0 = self.pipe.img2latent(self.imgA)
        latB0 = self.pipe.img2latent(self.imgB)
        
        # Get latent dimensions
        self.latent_shape = latA0.shape[1:]  # (C, H, W) without batch
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
    
    def _compute_semantic_term(
        self,
        X: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the semantic consistency regularization term.
        
        Args:
            X: Position tensor [N, D]
            V: Velocity tensor [N, D]
            
        Returns:
            term3: Semantic gradient tensor [N, D]
        """
        if self.semantic_reg is None:
            return torch.zeros_like(X)
        
        try:
            semantic_grad = self.semantic_reg.compute_gradient(
                X, self.pipe, V, self.latent_shape
            )
            
            # Project to orthogonal complement of velocity (same as other terms)
            semantic_grad = o_project_(semantic_grad, V)
            
            # Apply sphere constraint if enabled
            if self.sphere_constraint:
                semantic_grad = o_project_(semantic_grad, X)
            
            return semantic_grad * self.semantic_weight
            
        except Exception as e:
            print(f'[Warning] Semantic regularization failed: {e}')
            return torch.zeros_like(X)
    
    def bvp_gradient(
        self,
        X: torch.Tensor,
        V: torch.Tensor,
        A: torch.Tensor,
        t: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], float, float]:
        """
        Compute BVP gradient for geodesic optimization.
        
        Three terms:
            term1: Score distillation gradient - pushes path toward high probability density
            term2: Acceleration penalty - ensures geometric smoothness (geodesic equation)
            term3: Semantic consistency - ensures semantic continuity via DINO features
        
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
        lats = X.reshape(-1, *self.latent_shape)
        
        # Get embeddings
        prompt_embeds, pooled_embeds = self._get_embeddings_at_t(t)
        
        # =================================================================
        # Term 1: Probability density gradient via score distillation
        # This estimates ∇log p(γ) - pushes path toward high-density regions
        # =================================================================
        d_logps = self.score_dist.grad_compute_batch(lats, prompt_embeds, pooled_embeds)
        d_logps = d_logps.reshape(-1, self.latent_dim)
        
        # Project if using sphere constraint
        if self.sphere_constraint:
            d_logps = o_project_(d_logps, X)
            A = o_project_(A, X)
        
        # Compute geodesic equation terms
        V_norm2 = torch.sum(V * V, dim=-1)
        A_scaled = A / V_norm2[:, None]
        
        # Project to orthogonal complement of velocity: (I - γ̂˙γ̂˙ᵀ)
        term1 = o_project_(d_logps, V)
        
        # =================================================================
        # Term 2: Smoothness constraint (acceleration penalty)
        # This is γ̈/||γ̇||² from the Euler-Lagrange equation
        # Penalizes path curvature to ensure geometric smoothness
        # =================================================================
        term2 = o_project_(A_scaled, V) * (1 / self.alpha)
        
        # =================================================================
        # Term 3: Semantic consistency regularization
        # Penalizes jumps in DINO feature space between adjacent frames
        # =================================================================
        if self.use_semantic_reg and self.semantic_reg is not None:
            term3 = self._compute_semantic_term(X, V)
        else:
            term3 = torch.zeros_like(X)
        
        # =================================================================
        # Combined gradient: all three terms work together
        # =================================================================
        grad = -(term1 + term2 + term3)
        
        # Analyze gradients
        g_n, g1_n, g2_n, g3_n, g_angle = self.grad_analyzer.grad_analysis(
            t, self.cur_iter, term1, term2, term3, grad
        )
        
        # Heuristic: skip if acceleration term dominates probability term
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
            semantic_info = f', semantic={self.use_semantic_reg}' if self.use_semantic_reg else ''
            print(f'[BVP-Flux] iter={self.cur_iter}, t={t_str}, grad={g_n:.6f}, '
                  f'angle={g_angle:.1f}{semantic_info}')
        
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
            imgs.extend(self.io.backward_multi(
                X_out[1:-1], prompt_embeds[1:-1], pooled_embeds[1:-1], self.latent_shape
            ))
            imgs.append(self.imgB)
        else:
            imgs = self.io.backward_multi(X_out, prompt_embeds, pooled_embeds, self.latent_shape)
        
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
        if self.use_semantic_reg:
            print(f'[BVP-Flux] Semantic regularization weight: {self.semantic_weight}')
        
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
        
        # Save final velocities
        ts = torch.linspace(0, 1, 17, device=self.device)
        torch.save(self.spline(ts, 1), os.path.join(self.io.out_dir, 'final_vs.pt'))
        
        print('[BVP-Flux] Optimization complete')