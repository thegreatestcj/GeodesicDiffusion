"""
Flux BVP Solver Entry Point with Semantic Regularization

Usage:
    python test_bvp_flux.py --c configs/config_flux_semantic.yaml

Requirements:
    - GPU with 24GB+ VRAM (or enable CPU offload)
    - HuggingFace account with Flux access
    - Login: huggingface-cli login
"""

import os
import sys
import yaml
import shutil
import argparse
from PIL import Image
import torch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Flux BVP Solver with Semantic Regularization')
    parser.add_argument('--c', type=str, default='configs/config_flux_semantic.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use')
    parser.add_argument('--no-offload', action='store_true',
                        help='Disable CPU offload (requires more VRAM)')
    args = parser.parse_args()
    
    # Check CUDA
    if not torch.cuda.is_available():
        print('[ERROR] CUDA not available. Flux requires GPU.')
        sys.exit(1)
    
    # Print GPU info
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f'[INFO] GPU: {gpu_name}, Memory: {gpu_mem:.1f}GB')
    
    if gpu_mem < 20 and not args.no_offload:
        print('[WARNING] GPU memory < 20GB. CPU offload will be enabled.')
    
    # Load config
    print(f'[INFO] Loading config: {args.c}')
    with open(args.c, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup output directory
    out_dir = config['output_args']['out_dir']
    test_name = config['test_name']
    
    config['output_args']['out_dir'] = os.path.join(out_dir, test_name)
    config['tv_args']['tv_ckpt_folder'] = os.path.join(
        config['tv_args']['tv_ckpt_folder'], test_name
    )
    
    # Clean and create output directory
    out_path = config['output_args']['out_dir']
    if os.path.exists(out_path):
        print(f'[INFO] Removing existing output: {out_path}')
        shutil.rmtree(out_path)
    os.makedirs(out_path, exist_ok=True)
    os.makedirs(config['tv_args']['tv_ckpt_folder'], exist_ok=True)
    
    # Copy config to output
    shutil.copy(args.c, out_path)
    
    # Load Flux pipeline
    print('[INFO] Loading Flux pipeline...')
    print('[INFO] This may take several minutes for first download (~30GB)')
    
    from model.pipeline_flux import load_flux_pipe
    
    flux_args = config.get('flux_args', {})
    enable_offload = not args.no_offload and flux_args.get('enable_cpu_offload', True)
    
    pipe = load_flux_pipe(
        device=args.device,
        enable_cpu_offload=enable_offload,
    )
    
    # Load images
    print(f'[INFO] Loading images...')
    imgA = Image.open(config['pathA']).convert('RGB')
    imgB = Image.open(config['pathB']).convert('RGB')
    print(f'[INFO] Image A: {config["pathA"]} ({imgA.size})')
    print(f'[INFO] Image B: {config["pathB"]} ({imgB.size})')
    
    # Get semantic regularization args
    semantic_args = config.get('semantic_args', {})
    if semantic_args.get('use_semantic_reg', False):
        print(f'[INFO] Semantic regularization enabled:')
        print(f'       - weight: {semantic_args.get("semantic_weight", 0.1)}')
        print(f'       - model: {semantic_args.get("dino_model", "dinov2_vits14")}')
    else:
        print('[INFO] Semantic regularization disabled')
    
    # Create BVP solver
    print('[INFO] Creating BVP solver...')
    from model.bvp_flux import Geodesic_BVP_Flux
    
    bvp_solver = Geodesic_BVP_Flux(
        pipe=pipe,
        imgA=imgA,
        imgB=imgB,
        promptA=config['promptA'],
        promptB=config['promptB'],
        noise_level=config['noise_level'],
        alpha=config['alpha'],
        grad_args=config['grad_args'],
        bisect_args=config['bisect_args'],
        output_args=config['output_args'],
        tv_args=config['tv_args'],
        opt_args=config['opt_args'],
        semantic_args=semantic_args,  # Pass semantic regularization args
        use_lerp_cond=config.get('use_lerp_cond', True),
        sphere_constraint=config.get('sphere_constraint', True),
        grad_analysis_out=config.get('grad_analysis_out', True),
        test_name=test_name,
    )
    
    # Run optimization
    print('[INFO] Starting BVP optimization...')
    bvp_solver.solve()
    
    print('[INFO] Done!')
    print(f'[INFO] Results saved to: {out_path}')


if __name__ == '__main__':
    main()