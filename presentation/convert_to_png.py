"""
Convert numpy arrays to PNG using pypng (pure Python, no C dependencies)
"""
import numpy as np
from pathlib import Path
import struct

def write_png(filename, img_array):
    """
    Simple PNG writer for RGB images
    Uses pypng if available, otherwise saves as PPM (can be converted later)
    """
    try:
        import png
        height, width, channels = img_array.shape
        
        # Flatten to 2D array
        img_2d = img_array.reshape(-1, width * channels)
        
        with open(filename, 'wb') as f:
            writer = png.Writer(width=width, height=height, greyscale=False)
            writer.write(f, img_2d)
        return True
    except ImportError:
        # Fallback: save as PPM format (simple, can be converted later)
        ppm_filename = filename.replace('.png', '.ppm')
        height, width, channels = img_array.shape
        
        with open(ppm_filename, 'wb') as f:
            # PPM header
            f.write(f'P6\n{width} {height}\n255\n'.encode())
            # Write pixel data
            f.write(img_array.tobytes())
        
        print(f"   (Saved as PPM: {ppm_filename} - you can convert with imagemagick: convert {ppm_filename} {filename})")
        return False

# Try to install pypng first
print("Attempting to install pypng...")
import subprocess
try:
    subprocess.check_call(['pip3', 'install', '--user', 'pypng'], 
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✓ pypng installed")
except:
    print("⚠ Could not install pypng, will save as PPM format instead")

print("\nConverting images...")
dataset_dir = Path("dataset_images")

for npy_file in sorted(dataset_dir.glob('*.npy')):
    img = np.load(npy_file)
    png_file = dataset_dir / (npy_file.stem + '.png')
    
    if write_png(str(png_file), img):
        print(f"✓ {npy_file.name} → {png_file.name}")
    else:
        print(f"✓ {npy_file.name} → {npy_file.stem}.ppm")

print("\nDone!")

