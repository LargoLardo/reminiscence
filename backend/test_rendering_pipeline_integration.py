#!/usr/bin/env python3
"""
Integration test demonstrating the complete rendering pipeline workflow.
Shows how to prepare COLMAP data for FastGS training.
"""

from pathlib import Path
import sys
from rendering_pipeline import prepare_fastgs_input_and_train, PipelineResult


def test_pipeline_integration():
    """
    Demonstrates the complete pipeline workflow.
    
    To use with real COLMAP data:
    1. Generate COLMAP data using prepare_colmap_windows.py
    2. The output will be in backend/output with structure:
       - output/images/         (jpg/png images)
       - output/sparse/0/       (COLMAP binary files)
       - output/sparse_points.ply (point cloud)
    3. Call prepare_fastgs_input_and_train() as shown below
    4. The function will prepare the dataset for FastGS training
    """
    
    print("=" * 70)
    print("Integration Test: Complete Rendering Pipeline Workflow")
    print("=" * 70)
    
    backend_dir = Path(__file__).parent
    fastgs_root = backend_dir.parent / "fastgs"
    
    print(f"\nProject Structure:")
    print(f"  Backend: {backend_dir}")
    print(f"  FastGS:  {fastgs_root}")
    print(f"  Output:  {backend_dir / 'output'}")
    
    print("\n" + "-" * 70)
    print("Pipeline Workflow")
    print("-" * 70)
    
    print("""
1. INPUT: COLMAP Output Directory
   └─ output/
      ├─ images/                 (calibrated camera images)
      ├─ sparse/0/              (COLMAP reconstruction)
      │  ├─ cameras.bin         (camera intrinsic models)
      │  ├─ images.bin          (registered image poses)
      │  └─ points3D.bin        (sparse point cloud)
      └─ sparse_points.ply      (point cloud for visualization)

2. PROCESSING: prepare_fastgs_input_and_train()
   ├─ Validate COLMAP files exist
   ├─ Generate next dataset index (input_1, input_2, etc.)
   ├─ Create dataset directory structure
   ├─ Copy files to FastGS format
   ├─ Remove original COLMAP output (cleanup)
   └─ Build WSL training command

3. OUTPUT: FastGS Dataset
   └─ fastgs/datasets/input/input_N/
      ├─ images/                 (same as input)
      ├─ sparse/0/
      │  ├─ cameras.bin          (copied from COLMAP)
      │  ├─ images.bin           (copied from COLMAP)
      │  ├─ points3D.bin         (copied from COLMAP)
      │  ├─ points3d.bin         (alias for FastGS compat)
      │  └─ points3d.ply         (for validation)
      
4. TRAINING: Via WSL
   └─ Run FastGS training on the prepared dataset
      cd fastgs && python train.py -s ./datasets/input/input_N --eval
""")
    
    print("-" * 70)
    print("Code Example")
    print("-" * 70)
    
    example_code = '''
from pathlib import Path
from rendering_pipeline import prepare_fastgs_input_and_train

# Setup paths
backend_dir = Path("backend")
fastgs_root = Path("fastgs")
colmap_output = backend_dir / "output"

# Process COLMAP data for training
try:
    result = prepare_fastgs_input_and_train(
        colmap_output_dir=colmap_output,
        fastgs_root=fastgs_root,
        run_training=False  # Set to True to auto-start training via WSL
    )
    
    print(f"Dataset prepared: {result.dataset_name}")
    print(f"Dataset path: {result.dataset_path}")
    print(f"Model path: {result.model_path}")
    
    # If run_training=False, run training manually:
    # subprocess.run(["wsl", "bash", "-lc", result.wsl_command], check=True)
    
except Exception as e:
    print(f"Error: {e}")
'''
    
    print(example_code)
    
    print("-" * 70)
    print("Key Function Reference")
    print("-" * 70)
    
    print("""
prepare_fastgs_input_and_train(colmap_output_dir, fastgs_root, run_training=True)
    
    Args:
        colmap_output_dir: Path to COLMAP output directory
        fastgs_root: Path to FastGS repository root
        run_training: If True, automatically runs WSL training (requires WSL)
    
    Returns:
        PipelineResult with:
        - dataset_name: The generated dataset name (e.g., "input_5")
        - dataset_path: Full path to prepared dataset
        - model_path: Path where trained model will be saved
        - wsl_command: Complete WSL command for training
    
    Notes:
        - Requires COLMAP binary output at expected paths
        - Automatically generates next available dataset index
        - Deletes COLMAP output directory after processing (cleanup)
        - Requires WSL for training if run_training=True
""")
    
    print("\n" + "=" * 70)
    print("Test: Current Status")
    print("=" * 70)
    
    output_exists = (backend_dir / "output").exists()
    fastgs_exists = fastgs_root.exists()
    
    print(f"\n✓ COLMAP output directory exists: {output_exists}")
    print(f"✓ FastGS directory exists: {fastgs_exists}")
    
    if output_exists:
        output_size = sum(
            f.stat().st_size 
            for f in (backend_dir / "output").rglob("*") 
            if f.is_file()
        )
        print(f"  Size: {output_size / (1024*1024):.1f} MB")
        
        print("\n✓ Ready to process COLMAP data!")
        print("\n  Usage:")
        print("  python -c \"")
        print("  from pathlib import Path")
        print("  from rendering_pipeline import prepare_fastgs_input_and_train")
        print("  result = prepare_fastgs_input_and_train(")
        print("      Path('backend/output'),")
        print("      Path('fastgs'),")
        print("      run_training=False")
        print("  )\"")
    else:
        print("\n⚠ COLMAP output directory not found!")
        print("  Generate it using: python prepare_colmap_windows.py <video> output")
        print("  Or: python prepare_colmap.py <input_dir> output")
    
    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(test_pipeline_integration())
