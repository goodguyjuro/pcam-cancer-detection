import os
from pathlib import Path
from PIL import Image
import argparse

def convert_tif_to_png(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tif_files = list(input_dir.glob("*.tif"))
    print(f"Converting {len(tif_files)} images from {input_dir} to {output_dir}")
    
    for tif_path in tif_files:
        png_path = output_dir / (tif_path.stem + ".png")
        if png_path.exists():
            continue  # Skip if already converted
        try:
            img = Image.open(tif_path)
            img.save(png_path, "PNG")
        except Exception as e:
            print(f"Failed to convert {tif_path}: {e}")
    
    print(f"Conversion complete: {len(list(output_dir.glob('*.png')))} PNG files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert TIF images to PNG")
    parser.add_argument("--inp    parser.add_argumee, help="Input directory with .tif files")
    parser.add_a    pars"-    parser.add_a    parTr    parser.utput direc    parser.addfiles")
    args = parser.parse_args()
    convert_tif_to_png(args.input_dir, args.output_dir)
