"""
trim white spaces for gait 
"""
import os
from glob import glob
import fnmatch
from PIL import Image, ImageChops


def trim(filename):
   im = Image.open(filename)
   bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
   diff = ImageChops.difference(im, bg)
   diff = ImageChops.add(diff, diff, 2.0, -100)
   bbox = diff.getbbox()
   if bbox:
       return im.crop(bbox)

def main():
    print(os.getcwd())
    folder_path = f'data/frontview'
    suffix = '*.png'
    files = glob(f'{folder_path}/{suffix}')

    for f in files:
        trimmed_image = trim(f)
        trimmed_image.save(f"output/frontview/trimmed_{f.split('/')[-1]}")


def count_png_files(folder_path):
    png_counts = {}

    for root, dirs, files in os.walk(folder_path):
        png_count = len(fnmatch.filter(files, '*.png'))
        png_counts[root] = png_count

    return png_counts

def gait_png_count():
    folder_path = '/home/qiyuan/2023fall/Gait/gait_dataset_keypoints/output_png_folder'
    subfolder_png_counts = count_png_files(folder_path)

    for subfolder, count in subfolder_png_counts.items():
        print(f'The number of PNG files in {subfolder} is: {count}')


if __name__ == '__main__':
    # main()
    gait_png_count()
