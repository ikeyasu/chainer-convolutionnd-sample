from PIL import Image
import numpy as np
import argparse


def main(path, out_path):
  img = Image.open(path, 'r')
  h, w = img.size
  cw = h // 2
  cx = int((w - cw) / 4)
  cy = int((h - cw) / 4)
  clp = img.crop((cx, cy, w - cx, h - cy))
  resized = clp.resize((32, 32))
  resized.save(out_path, 'JPEG', quality=100, optimize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='clipping and resize')
    parser.add_argument('--input', '-i', default='input.jpg',
                        help='input file')
    parser.add_argument('--output', '-o', default='output.jpg',
                        help='output file')
    args = parser.parse_args()
    main(args.input, args.output)
