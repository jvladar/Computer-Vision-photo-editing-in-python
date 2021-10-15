
import scipy.stats as st
import math
import copy
import numpy as np

def normalize_kernel(kernel : np.ndarray) -> np.ndarray :
    """ Normalizes the kernel. Returns original kernel if the sum of elements is 0."""
    sum = kernel.sum()
    if sum == 0 :
        return kernel
    else:
        return kernel/sum


"""
From https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
"""
def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel."""

    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()


"""
Function to rotate an image around its center and using an appropriate interpolation funciton with signature
defined in hw1.py.
"""
def rotate_image (interpolation_fn, image : np.ndarray, rotation_angle : float, in_place : bool = False) -> np.ndarray :
    radians = math.radians(rotation_angle)
    image_copy = np.zeros_like(image)
    image_height, image_width, *_ = image.shape
    for r in range(image_height):
        for c in range(image_width):
            x0 = c - image_width/2.0
            y0 = r - image_height/2.0
            x1 = x0 * math.cos(radians) - y0 * math.sin(radians)
            y1 = x0 * math.sin(radians) + y0 * math.cos(radians)
            x1 += image_width/2.0
            y1 += image_height/2.0
            rgb = interpolation_fn (image, x1, y1)
            image_copy[r][c] = rgb
    return image_copy

def rotate_image_fast(interpolation_fn, image: np.ndarray, rotation_angle: float, in_place: bool = False) -> np.ndarray:
    """
    Function to rotate an image around its center and using an appropriate interpolation function with signature
    defined in hw1.py. Optimisations by A. Käver.
    """
    radians = math.radians(rotation_angle)
    image_copy = np.zeros_like(image)
    image_height, image_width, *_ = image.shape
    image_height_div2 = image_height / 2.0
    image_width_div2 = image_width / 2.0

    cos = math.cos(radians)
    sin = math.sin(radians)

    for r in range(image_height):
        x0 = r - image_height_div2
        x0_cos = x0 * cos
        x0_sin = x0 * sin
        for c in range(image_width):
            y0 = c - image_width_div2
            x1 = x0_cos - y0 * sin
            y1 = x0_sin + y0 * cos
            x1 += image_height_div2
            y1 += image_width_div2
            rgb = interpolation_fn(image, x1, y1)
            image_copy[r][c] = rgb
    return image_copy
