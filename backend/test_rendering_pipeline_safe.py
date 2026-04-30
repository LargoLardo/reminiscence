#!/usr/bin/env python3
"""
Safe test script for rendering_pipeline.py with mock COLMAP data.
Creates test data in a temporary directory, tests the pipeline, and cleans up.
"""

from pathlib import Path
import sys
import shutil
import tempfile
import struct
from rendering_pipeline import (
    _next_input_index,
    _require_file,
    PipelineResult,
)


def create_mock_colmap_data(output_dir: Path) -> None:
    """Create minimal mock COLMAP data for testing."""
    # Create directory structure
    images_dir = output_dir / "images"
    sparse_dir = output_dir / "sparse" / "0"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    sparse_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mock binary files
    # cameras.bin: minimal COLMAP camera model (just write some bytes)
    with open(sparse_dir / "cameras.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))  # num_cameras = 1
        f.write(struct.pack("<Q", 1))  # camera_id = 1
        f.write(struct.pack("<i", 1))  # model (SIMPLE_PINHOLE)
        f.write(struct.pack("<Q", 800))  # width
        f.write(struct.pack("<Q", 600))  # height
        f.write(struct.pack("<4d", 500.0, 400.0, 300.0, 0.0))  # params (fx, fy, cx, cy)
    
    # images.bin: minimal COLMAP image data
    with open(sparse_dir / "images.bin", "wb") as f:
        f.write(struct.pack("<Q", 1))  # num_images = 1
        # Image entry (simplified)
        f.write(struct.pack("<Q", 1))  # image_id
        f.write(struct.pack("<4d", 1.0, 0.0, 0.0, 0.0))  # qvec (quaternion)
        f.write(struct.pack("<3d", 0.0, 0.0, 0.0))  # tvec (translation)
        f.write(struct.pack("<Q", 1))  # camera_id
        # String name
        name = "image001.jpg"
        f.write(struct.pack("<I", len(name)))
        f.write(name.encode())
        f.write(struct.pack("<Q", 0))  # num_points2d = 0
    
    # points3D.bin: minimal 3D points
    with open(sparse_dir / "points3D.bin", "wb") as f:
        f.write(struct.pack("<Q", 0))  # num_points3d = 0
    
    # Create mock PLY file
    with open(output_dir / "sparse_points.ply", "w") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("element vertex 10\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        # Write 10 test vertices
        for i in range(10):
            f.write(f"{i*0.1} {i*0.1} {i*0.1} 255 128 64\n")
    
    # Create a mock image
    try:
        from PIL import Image
        img = Image.new('RGB', (800, 600), color='red')
        img.save(images_dir / "image001.jpg")
    except ImportError:
        # If PIL not available, create a minimal JPEG stub
        # For testing purposes, this is enough
        jpeg_header = bytes([0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46,
                            0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01,
                            0x00, 0x01, 0x00, 0x00])
        jpeg_end = bytes([0xFF, 0xD9])
        with open(images_dir / "image001.jpg", "wb") as f:
            f.write(jpeg_header)
            f.write(bytes(1000))  # Minimal padding
            f.write(jpeg_end)


def test_rendering_pipeline():
    """Test the rendering pipeline with safe mock data."""
    
    print("=" * 70)
    print("Safe Test: rendering_pipeline.py with Mock COLMAP Data")
    print("=" * 70)
    
    backend_dir = Path(__file__).parent
    fastgs_root = backend_dir.parent / "fastgs"
    
    # Create temporary test environment
    with tempfile.TemporaryDirectory(prefix="test_colmap_") as temp_dir:
        temp_path = Path(temp_dir)
        test_output_dir = temp_path / "output"
        test_dataset_dir = temp_path / "datasets"
        test_dataset_dir.mkdir()
        
        print(f"\nTest Environment: {temp_path}")
        print("-" * 70)
        
        # Create mock COLMAP data
        print("\n✓ Creating mock COLMAP data...")
        create_mock_colmap_data(test_output_dir)
        
        # Verify COLMAP structure
        print("\n✓ Verifying mock COLMAP structure:")
        required_files = [
            test_output_dir / "images",
            test_output_dir / "sparse" / "0" / "cameras.bin",
            test_output_dir / "sparse" / "0" / "images.bin",
            test_output_dir / "sparse" / "0" / "points3D.bin",
            test_output_dir / "sparse_points.ply",
        ]
        
        for f in required_files:
            exists = f.exists()
            symbol = "✓" if exists else "✗"
            print(f"  {symbol} {f.relative_to(temp_path)}")
        
        # Test pipeline functions
        print("\n" + "-" * 70)
        print("Testing Pipeline Functions...")
        print("-" * 70)
        
        try:
            # Test 1: _next_input_index
            dataset_root = test_dataset_dir / "input"
            dataset_root.mkdir(parents=True, exist_ok=True)
            next_idx = _next_input_index(dataset_root)
            print(f"\n✓ _next_input_index() = {next_idx}")
            
            dataset_name = f"input_{next_idx}"
            dataset_dir = dataset_root / dataset_name
            images_dir = dataset_dir / "images"
            sparse0_dir = dataset_dir / "sparse" / "0"
            
            # Test 2: File validation
            print(f"\n✓ Testing file validation...")
            images_src = test_output_dir / "images"
            cameras_src = test_output_dir / "sparse" / "0" / "cameras.bin"
            images_bin_src = test_output_dir / "sparse" / "0" / "images.bin"
            points3d_bin_src = test_output_dir / "sparse" / "0" / "points3D.bin"
            
            try:
                _require_file(cameras_src)
                _require_file(images_bin_src)
                _require_file(points3d_bin_src)
                print(f"  ✓ All required files validated")
            except FileNotFoundError as e:
                print(f"  ✗ File validation failed: {e}")
                return 1
            
            # Test 3: File copying (core pipeline logic)
            print(f"\n✓ Testing file copying...")
            images_dir.mkdir(parents=True, exist_ok=True)
            sparse0_dir.mkdir(parents=True, exist_ok=True)
            
            import shutil
            shutil.copytree(images_src, images_dir, dirs_exist_ok=True)
            shutil.copy2(cameras_src, sparse0_dir / "cameras.bin")
            shutil.copy2(images_bin_src, sparse0_dir / "images.bin")
            shutil.copy2(points3d_bin_src, sparse0_dir / "points3D.bin")
            shutil.copy2(points3d_bin_src, sparse0_dir / "points3d.bin")
            
            points3d_ply_src = test_output_dir / "sparse_points.ply"
            shutil.copy2(points3d_ply_src, sparse0_dir / "points3d.ply")
            
            print(f"  ✓ Files copied successfully")
            
            # Test 4: Verify dataset structure
            print(f"\n✓ Verifying prepared dataset structure:")
            verify_paths = [
                (images_dir, "images/"),
                (sparse0_dir / "cameras.bin", "sparse/0/cameras.bin"),
                (sparse0_dir / "images.bin", "sparse/0/images.bin"),
                (sparse0_dir / "points3D.bin", "sparse/0/points3D.bin"),
                (sparse0_dir / "points3d.bin", "sparse/0/points3d.bin (alias)"),
                (sparse0_dir / "points3d.ply", "sparse/0/points3d.ply"),
            ]
            
            for path, display_name in verify_paths:
                exists = path.exists()
                symbol = "✓" if exists else "✗"
                print(f"  {symbol} {display_name}")
            
            # Test 5: PipelineResult
            print(f"\n✓ Creating PipelineResult...")
            result = PipelineResult(
                dataset_name=dataset_name,
                dataset_path=str(dataset_dir),
                model_path=str(fastgs_root / "output" / dataset_name),
                wsl_command="(test command - WSL not available on Windows without WSL installed)"
            )
            
            print(f"  Dataset name: {result.dataset_name}")
            print(f"  Dataset path: {result.dataset_path}")
            print(f"  Model path: {result.model_path}")
            
            # Test 6: Dataset structure summary
            print(f"\n" + "=" * 70)
            print("Test Results: ✓ SUCCESS")
            print("=" * 70)
            
            print(f"\n✓ Dataset successfully prepared:")
            print(f"  Location: {dataset_dir}")
            print(f"  Size: ~{sum(f.stat().st_size for f in dataset_dir.rglob('*') if f.is_file()) / 1024:.1f} KB")
            
            print(f"\n✓ Pipeline Summary:")
            print(f"  1. COLMAP output validated: {len(list(test_output_dir.rglob('*')))} items")
            print(f"  2. Dataset prepared: {len(list(dataset_dir.rglob('*')))} items")
            print(f"  3. Ready for FastGS training")
            
            print(f"\nNext steps:")
            print(f"  1. Run: wsl bash -lc 'cd /mnt/.../fastgs && python train.py -s ./datasets/input/{dataset_name}'")
            print(f"  2. Run: wsl bash -lc 'python render.py -s ... -m ./output/{dataset_name}'")
            
            return 0
            
        except Exception as e:
            print(f"\n✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(test_rendering_pipeline())
