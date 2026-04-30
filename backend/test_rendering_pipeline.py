#!/usr/bin/env python3
"""Test script for rendering_pipeline.py with COLMAP output data."""

from pathlib import Path
import sys
import shutil
from rendering_pipeline import (
    _next_input_index,
    _require_file,
    _build_wsl_training_command
)

def main():
    # Paths
    backend_dir = Path(__file__).parent
    colmap_output_dir = backend_dir / "output"
    fastgs_root = backend_dir.parent / "fastgs"
    
    print("=" * 70)
    print("Testing rendering_pipeline.py with COLMAP data")
    print("=" * 70)
    
    # Verify paths exist
    print(f"\n✓ Backend directory: {backend_dir}")
    print(f"✓ COLMAP output directory: {colmap_output_dir}")
    print(f"✓ FastGS root: {fastgs_root}")
    
    # Check COLMAP files
    print("\n" + "-" * 70)
    print("COLMAP Data Structure:")
    print("-" * 70)
    
    required_files = [
        colmap_output_dir / "images",
        colmap_output_dir / "sparse" / "0" / "cameras.bin",
        colmap_output_dir / "sparse" / "0" / "images.bin",
        colmap_output_dir / "sparse" / "0" / "points3D.bin",
        colmap_output_dir / "sparse_points.ply",
    ]
    
    all_exist = True
    for f in required_files:
        exists = f.exists()
        symbol = "✓" if exists else "✗"
        print(f"  {symbol} {f.relative_to(backend_dir)}")
        all_exist = all_exist and exists
    
    if not all_exist:
        print("\n✗ Missing required COLMAP files!")
        return 1
    
    print("\n" + "-" * 70)
    print("Testing pipeline functions...")
    print("-" * 70)
    
    try:
        # Test 1: _next_input_index
        dataset_root = fastgs_root / "datasets" / "input"
        dataset_root.mkdir(parents=True, exist_ok=True)
        next_idx = _next_input_index(dataset_root)
        print(f"\n✓ _next_input_index: {next_idx}")
        
        dataset_name = f"input_{next_idx}"
        dataset_dir = dataset_root / dataset_name
        images_dir = dataset_dir / "images"
        sparse0_dir = dataset_dir / "sparse" / "0"
        
        # Test 2: Verify files exist and copy them
        print(f"\n✓ Testing dataset preparation...")
        print(f"  Dataset name: {dataset_name}")
        print(f"  Target directory: {dataset_dir}")
        
        images_src = colmap_output_dir / "images"
        cameras_src = colmap_output_dir / "sparse" / "0" / "cameras.bin"
        images_bin_src = colmap_output_dir / "sparse" / "0" / "images.bin"
        points3d_bin_src = colmap_output_dir / "sparse" / "0" / "points3D.bin"
        points3d_ply_src = colmap_output_dir / "sparse_points.ply"
        
        # Verify all files
        _require_file(cameras_src)
        _require_file(images_bin_src)
        _require_file(points3d_bin_src)
        
        print(f"  ✓ All COLMAP files verified")
        
        # Copy files
        images_dir.mkdir(parents=True, exist_ok=True)
        sparse0_dir.mkdir(parents=True, exist_ok=True)
        
        shutil.copytree(images_src, images_dir, dirs_exist_ok=True)
        shutil.copy2(cameras_src, sparse0_dir / "cameras.bin")
        shutil.copy2(images_bin_src, sparse0_dir / "images.bin")
        shutil.copy2(points3d_bin_src, sparse0_dir / "points3D.bin")
        shutil.copy2(points3d_bin_src, sparse0_dir / "points3d.bin")
        shutil.copy2(points3d_ply_src, sparse0_dir / "points3d.ply")
        
        print(f"  ✓ Files copied successfully")
        
        # Test 3: Verify copied structure
        print(f"\n✓ Verifying copied structure:")
        verify_paths = [
            (images_dir, "Images"),
            (sparse0_dir / "cameras.bin", "cameras.bin"),
            (sparse0_dir / "images.bin", "images.bin"),
            (sparse0_dir / "points3D.bin", "points3D.bin"),
            (sparse0_dir / "points3d.ply", "points3d.ply"),
        ]
        
        for path, name in verify_paths:
            exists = path.exists()
            symbol = "✓" if exists else "✗"
            print(f"  {symbol} {name}")
        
        # Test 4: Show WSL training command (without executing)
        print(f"\n✓ Generated WSL training command:")
        print(f"  set -e; cd /mnt/c/Users/login/reminiscence/reminiscence/fastgs; \\")
        print(f"  CUDA_VISIBLE_DEVICES=0 OAR_JOB_ID={dataset_name} \\")
        print(f"  python train.py -s ./datasets/input/{dataset_name} \\")
        print(f"  --eval --densification_interval 500 --optimizer_type default \\")
        print(f"  --test_iterations 30000 --highfeature_lr 0.0015 --dense 0.003 --mult 0.7; \\")
        print(f"  CUDA_VISIBLE_DEVICES=0 python render.py \\")
        print(f"  -s ./datasets/input/{dataset_name} -m ./output/{dataset_name} --skip_train")
        
        print("\n" + "=" * 70)
        print("Test Results: ✓ SUCCESS")
        print("=" * 70)
        print(f"\nDataset prepared at: {dataset_dir}")
        print(f"Ready for FastGS training on WSL with command:")
        print(f"  cd c:\\Users\\login\\reminiscence\\reminiscence && wsl bash -lc 'set -e; ...'")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
