"""
Geodesic BVP Solver with Semantic Regularization

Based on: "Probability Density Geodesics in Image Diffusion Latent Space"
(Yu et al., CVPR 2025)

Modifications:
- Added SemanticRegularizer class for DINO-v2 based semantic consistency
- Extended bvp_gradient to include semantic regularization term (term3)
"""

from model.score import *
import os
from model.scheduler import *
from model.bisection import *
from model.text_inv import *
from model.io import *
from model.spline import *

import torch.nn.functional as F
from torchvision import transforms


# =============================================================================
# Semantic Regularizer
# =============================================================================

class SemanticRegularizer:
    """
    Computes semantic consistency regularization using DINO-v2 features.
    
    Encourages the interpolation path to maintain semantic coherence between
    adjacent frames by penalizing large jumps in DINO feature space.
    """
    
    def __init__(self, device='cuda', model_name='dinov2_vits14'):
        """
        Args:
            device: Torch device ('cuda' or 'cpu')
            model_name: DINO model variant ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14')
        """
        self.device = device
        self.model_name = model_name
        
        print(f'[SemanticReg] Loading {model_name}...')
        self.dino = torch.hub.load('facebookresearch/dinov2', model_name)
        self.dino.eval()
        self.dino.to(device)
        
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
        
        print(f'[SemanticReg] {model_name} loaded successfully.')
    
    def extract_features(self, images):
        """
        Extract DINO-v2 CLS token features from images.
        
        Args:
            images: Tensor of shape [B, 3, H, W] in range [0, 1]
            
        Returns:
            features: Tensor of shape [B, feature_dim]
        """
        images = torch.clamp(images, 0, 1)
        images_preprocessed = self.preprocess(images)
        
        with torch.no_grad():
            features = self.dino(images_preprocessed)
        
        return features
    
    def compute_gradient(self, latents, pipe, V):
        """
        Compute semantic regularization gradient.
        
        Args:
            latents: Tensor of shape [N, 16384] - flattened latent codes
            pipe: SimpleDiffusionPipeline for VAE decoding
            V: Tensor of shape [N, 16384] - velocity vectors
            
        Returns:
            grad: Tensor of shape [N, 16384] - semantic regularization gradient
        """
        N = latents.shape[0]
        
        if N < 2:
            return torch.zeros_like(latents)
        
        latents_4d = latents.reshape(N, 4, 64, 64)
        
        # Decode latents to images
        with torch.no_grad():
            scaled_latents = latents_4d / pipe.vae.config.scaling_factor
            
            images_list = []
            batch_size = 4
            for i in range(0, N, batch_size):
                batch = scaled_latents[i:i+batch_size]
                decoded = pipe.vae.decode(batch)['sample']
                images_list.append(decoded)
            
            images = torch.cat(images_list, dim=0)  # [N, 3, 512, 512]
            images = (images + 1) / 2  # Convert from [-1, 1] to [0, 1]
            
            features = self.extract_features(images)  # [N, feat_dim]
        
        # Compute feature differences between adjacent frames
        feat_diff = features[1:] - features[:-1]  # [N-1, feat_dim]
        
        # Build gradient: push each point toward its neighbors in feature space
        grad_features = torch.zeros_like(features)
        grad_features[1:] += feat_diff      # Contribution from left neighbor
        grad_features[:-1] -= feat_diff     # Contribution from right neighbor
        
        # Project to latent space
        grad_latents = self._project_to_latent(grad_features, latents.device)
        
        return grad_latents
    
    def _project_to_latent(self, grad_features, device):
        """
        Project feature-space gradient to latent space using random projection.
        
        Args:
            grad_features: Tensor of shape [N, feat_dim]
            device: Torch device
            
        Returns:
            grad_latents: Tensor of shape [N, 16384]
        """
        N, feat_dim = grad_features.shape
        latent_dim = 16384  # 4 * 64 * 64
        
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

class BVP_Optimiser():
    """Base class for BVP optimization with learning rate scheduling."""
    
    def __init__(self, iter_num, lr, lr_scheduler='constant', lr_divide=True):
        self.iter_num = iter_num
        self.lr_ini = lr
        self.lr_scheduler = lr_scheduler
        self.lr_divide = lr_divide

    def get_learning_rate(self, cur_iter, t):
        if self.lr_scheduler == 'constant':
            return self.lr_ini
        
        scale = cur_iter / self.iter_num
        
        if self.lr_scheduler == 'linear':
            cur_lr = self.lr_ini * (1 - scale)
        elif self.lr_scheduler == 'cosine':
            cur_lr = self.lr_ini * 0.5 * (1 + torch.cos(torch.pi * scale))
        elif self.lr_scheduler == 'polynomial':
            cur_lr = self.lr_ini * (1 - scale) ** 2
        else:
            raise ValueError(f'lr_scheduler not recognized: {self.lr_scheduler}')
        
        if self.lr_divide:
            cur_lr = cur_lr / len(t)
        
        return cur_lr


# =============================================================================
# Main Geodesic BVP Solver
# =============================================================================

class Geodesic_BVP(Score_Distillation, 
                   BVP_Optimiser,
                   Bisection_sampler, 
                   TextInversion, 
                   BVP_IO, 
                   Ouput_Grad):
    """
    Geodesic Boundary Value Problem solver with semantic regularization.
    
    The gradient consists of three terms:
        term1: Probability density gradient (via score distillation)
        term2: Smoothness constraint (acceleration penalty from geodesic equation)
        term3: Semantic consistency (DINO feature continuity) [NEW]
    """
    
    def __init__(self, 
            pipe: SimpleDiffusionPipeline, 
            imgA, 
            imgB, 
            promptA,
            promptB, 
            noise_level, 
            alpha, 
            grad_args,
            bisect_args,
            output_args,
            tv_args,
            opt_args,
            semantic_args=None,
            analysis_args=None,
            spline_type='spherical_cubic', 
            grad_analysis_out=True, 
            use_lerp_cond=False, 
            sphere_constraint=True, 
            test_name='test_bvp', 
            **kwargs
            ):
        self.imgA = imgA
        self.imgB = imgB
        self.promptA = promptA
        self.promptB = promptB
        self.test_name = test_name
        self.alpha = alpha
        self.time_step = pipe.get_t(noise_level, return_single=True)
        self.sphere_constraint = sphere_constraint
        self.use_lerp_cond = use_lerp_cond
        
        # Initialize parent classes
        Score_Distillation.__init__(self, pipe=pipe, time_step=self.time_step, **grad_args)
        BVP_Optimiser.__init__(self, **opt_args)
        Bisection_sampler.__init__(self, **bisect_args)
        TextInversion.__init__(self, pipe=pipe, **tv_args)
        BVP_IO.__init__(self, pipe=pipe, noise_level=noise_level, imgA=imgA, imgB=imgB, use_lerp_cond=use_lerp_cond, **output_args)
        Ouput_Grad.__init__(self, grad_analysis_out, os.path.join(self.out_dir, 'grad_check.txt'))

        # Optimization state
        self.cur_iter = 0
        self.path = dict()

        # Initialize conditional embeddings via text inversion
        latA0 = self.pipe.img2latent(self.imgA)
        latB0 = self.pipe.img2latent(self.imgB)
        
        if use_lerp_cond:
            self.embed_condA = self.text_inversion_load(self.promptA, latA0, self.test_name, 'A')
            self.embed_condB = self.text_inversion_load(self.promptB, latB0, self.test_name, 'B')
        else:
            prompt = self.promptA + ' ' + self.promptB
            lat0 = torch.cat([latA0, latB0], dim=0)
            self.embed_condAB = self.text_inversion_load(prompt, lat0, self.test_name, 'AB')

        # Initialize spline with great circle between endpoints
        if use_lerp_cond:
            latA = self.forward_single(latA0, self.embed_condA)
            latB = self.forward_single(latB0, self.embed_condB)
        else:
            latA = self.forward_single(latA0, self.embed_condAB)
            latB = self.forward_single(latB0, self.embed_condAB)
        
        pointA = latA.reshape(-1)
        pointB = latB.reshape(-1)
        
        if self.sphere_constraint:
            self.radius = 0.5 * (torch.norm(pointA) + torch.norm(pointB))
            pointA = norm_fix(pointA, self.radius)
            pointB = norm_fix(pointB, self.radius)
        
        self.spline = Spline(spline_type)
        self.spline.fit_spline(torch.tensor([0.0, 1.0]).to(self.device), torch.stack([pointA, pointB], dim=0))
        self.path[0] = pointA
        self.path[1] = pointB
        
        # =====================================================================
        # NEW: Initialize semantic regularization
        # =====================================================================
        if semantic_args is None:
            semantic_args = {}
        
        # Support both semantic_args dict and kwargs for backward compatibility
        self.use_semantic_reg = semantic_args.get('use_semantic_reg', kwargs.get('use_semantic_reg', False))
        self.semantic_weight = semantic_args.get('semantic_weight', kwargs.get('semantic_weight', 0.1))
        dino_model = semantic_args.get('dino_model', 'dinov2_vits14')
        
        if self.use_semantic_reg:
            print(f'[Geodesic_BVP] Semantic regularization enabled, weight={self.semantic_weight}')
            self.semantic_reg = SemanticRegularizer(device=self.device, model_name=dino_model)
        else:
            print(f'[Geodesic_BVP] Semantic regularization disabled')
            self.semantic_reg = None

    def bvp_gradient(self, X, V, A, t):
        """
        Compute the BVP gradient for path optimization.
        
        Three terms (all three are preserved):
            term1: Score distillation gradient - pushes path toward high probability density
            term2: Acceleration penalty - ensures geometric smoothness (from geodesic equation)
            term3: Semantic consistency - ensures semantic continuity via DINO features [NEW]
        
        Args:
            X: Position tensor [N, 16384] - points on the path
            V: Velocity tensor [N, 16384] - first derivative of spline
            A: Acceleration tensor [N, 16384] - second derivative of spline
            t: Parameter values [N] - position along path [0, 1]
            
        Returns:
            grad: Gradient tensor [N, 16384] or None if skipping
            g_n: Gradient norm (for logging)
            g_angle: Angle between gradient components (for logging)
        """
        lats = X.reshape(-1, 4, 64, 64)
        
        if self.use_lerp_cond:
            embed_cond = lerp_cond_embed(t, self.embed_condA, self.embed_condB)
        else:
            embed_cond = self.embed_condAB.repeat(lats.shape[0], 1, 1)
        
        # =================================================================
        # Term 1: Probability density gradient via score distillation
        # This estimates ∇log p(γ) - pushes path toward high-density regions
        # =================================================================
        d_logps = self.grad_compute_batch(lats, embed_cond)
        d_logps = d_logps.reshape(-1, 16384)
        
        if self.sphere_constraint:
            d_logps = o_project_(d_logps, X)
            A = o_project_(A, X)
        
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
        # Term 3: Semantic consistency regularization [NEW]
        # Penalizes jumps in DINO feature space between adjacent frames
        # =================================================================
        if self.use_semantic_reg and self.semantic_reg is not None:
            term3 = self._compute_semantic_term(X, V)
        else:
            term3 = 0
        
        # =================================================================
        # Combined gradient: all three terms work together
        # =================================================================
        grad = -(term1 + term2 + term3)
        
        # Gradient analysis for logging
        g_n, g1_n, g2_n, g_angle = self.grad_analysis(t, self.cur_iter, term1, term2, grad)

        # Heuristic: skip step if acceleration term dominates probability term
        if g1_n < g2_n:
            return None, g_n, g_angle
        
        return grad, g_n, g_angle
    
    def _compute_semantic_term(self, X, V):
        """
        Compute the semantic consistency regularization term.
        
        Args:
            X: Position tensor [N, 16384]
            V: Velocity tensor [N, 16384]
            
        Returns:
            term3: Semantic gradient tensor [N, 16384]
        """
        try:
            semantic_grad = self.semantic_reg.compute_gradient(X, self.pipe, V)
            
            # Project to orthogonal complement of velocity (same as other terms)
            semantic_grad = o_project_(semantic_grad, V)
            
            # Apply sphere constraint if enabled
            if self.sphere_constraint:
                semantic_grad = o_project_(semantic_grad, X)
            
            return semantic_grad * self.semantic_weight
            
        except Exception as e:
            print(f'[Warning] Semantic regularization failed: {e}')
            return torch.zeros_like(X)

    def step(self):
        """Perform one optimization step."""
        t_opt = self.get_control_t()
        if t_opt is None:
            return True  # Optimization finished
        
        t_opt = t_opt.to(self.device)
        
        # Get spline values at control points
        X_opt = self.spline(t_opt)
        V_opt = self.spline(t_opt, 1)
        A_opt = self.spline(t_opt, 2)
        
        # Compute gradient
        grad, g_n, g_angle = self.bvp_gradient(X_opt, V_opt, A_opt, t_opt)
        
        if grad is None:
            self.add_strength(None)
            self.cur_iter += 1
            return False
        
        cur_lr = self.get_learning_rate(self.cur_iter, t_opt)
        control_t = t_opt.detach().cpu().numpy()
        
        if self.cur_iter % 5 == 0:
            print('optimise {} t={} iteration: {}, grad_norm: {}, angle: {}'.format(
                self.test_name, control_t, self.cur_iter, g_n, g_angle))
        
        # Gradient descent update
        X_opt = X_opt - cur_lr * grad
        
        # Project back to sphere if using sphere constraint
        if self.sphere_constraint:
            X_opt = norm_fix_(X_opt, torch.tensor([self.radius] * X_opt.shape[0]).to(self.device))
        
        self.cur_iter += 1
        
        # Update path dictionary
        for i, t in enumerate(t_opt):
            self.path[t.item()] = X_opt[i]
        
        # Refit spline to updated control points
        t_fit = torch.tensor(sorted(self.path.keys())).to(self.device)
        X_fit = torch.stack([self.path[t.item()] for t in t_fit], dim=0)
        self.spline.fit_spline(t_fit, X_fit)

        self.add_strength(self.cur_iter)
        return False
       
    def solve(self):
        """Main solving loop for BVP optimization."""
        if self.use_lerp_cond:
            embed_cond_args = {'embed_cond_A': self.embed_condA, 'embed_cond_B': self.embed_condB}
        else:
            embed_cond_args = {'embed_cond': self.embed_condAB}
        
        self.output_bvp_sequence_if_need(0, self.spline, 'start', **embed_cond_args)
        
        for i in range(self.iter_num):
            finish = self.step()
            
            if finish or i == self.iter_num - 1:
                self.output_bvp_sequence_if_need(self.cur_iter, self.spline, 'final', **embed_cond_args)
                break
            else:
                self.output_bvp_sequence_if_need(self.cur_iter, self.spline, str(self.cur_iter), **embed_cond_args)
            
            torch.cuda.empty_cache()
        
        self.save_opt_points_if_need(self.path)
        
        ts = torch.linspace(0, 1, 17, device=self.device)
        torch.save(self.spline(ts, 1), os.path.join(self.out_dir, 'final_vs.pt'))