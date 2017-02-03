import math
import os
import subprocess
from PIL import Image

from . import util


def combine_plots_into_one(all_val_imgs, path_out, out_fname, imgs_in_row=10.):
    """
    Concatenate all images to one giant image
    Args:
        all_val_imgs:
        path_out:
        out_fname:
        imgs_in_row:

    Returns:

    """
    width, height = Image.open(all_val_imgs[0]).size
    png_info = Image.open(all_val_imgs[0]).info
    max_size = max(width, height)

    # assuming that all images have same size
    new_im = Image.new('RGB', (int(max_size * imgs_in_row), int(max_size * math.ceil(len(all_val_imgs)/imgs_in_row))))

    for idx, img in enumerate(all_val_imgs):
        x_pos = int(width * (idx % int(imgs_in_row)))  # upper left of image
        y_pos = int(height * (idx / int(imgs_in_row)))  # upper top of image

        im = Image.open(img)
        new_im.paste(im, (x_pos, y_pos))

    new_im.save(path_out + os.sep + out_fname + '.png', **png_info)


def make_movie(list_images, out_path, out_fname):
    """
    :param list_images:
    :param out_path:
    :param out_fname:
    :return:
    """
    util.make_dir_if_missing(out_path)

    convert_cmd = 'convert -delay 50 -loop 1 '+' '.join(list_images) + ' ' + out_path + os.sep + out_fname
    subprocess.call(convert_cmd, shell=True)
