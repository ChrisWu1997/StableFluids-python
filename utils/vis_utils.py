import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm
from PIL import Image
import os
from scipy.special import erf
from .math_utils import compute_curl


def draw_velocity(arr: np.ndarray, save_path=None, figsize=(4, 4)):
    """draw 2D velocity field"""
    if arr.shape[0] > 32:
        s = arr.shape[0] // 32 + 1
        arr = arr[::s, ::s]
    fig, ax = plt.subplots(figsize=figsize)
    indices = np.indices(arr.shape[:-1])
    ax.quiver(indices[0], indices[1], arr[..., 0], arr[..., 1], 
        scale=arr.shape[0], scale_units='width')
    fig.tight_layout()
    if save_path is not None:
        save_figure(fig, save_path, axis_off=True)
    return fig


def draw_curl(curl: np.ndarray, save_path=None):
    """draw 2D curl(vorticity) field"""
    curl = (erf(curl) + 1) / 2 # map range to 0~1
    img = cm.bwr(curl)
    img = Image.fromarray((img * 255).astype('uint8'))
    if save_path is not None:
        img.save(save_path)
    return img


def draw_density(density: np.ndarray, save_path=None):
    """draw 2D density field"""
    density = erf(np.clip(density, 0, None) * 2)
    img = cm.cividis(density)
    img = Image.fromarray((img * 255).astype('uint8'))
    if save_path is not None:
        img.save(save_path)
    return img


def draw_mix(curl: np.ndarray, density: np.ndarray, save_path=None):
    """R: curl, G: 1, B: density"""
    curl = (erf(curl) + 2) / 4
    img = np.dstack((curl, np.ones_like(curl), density))
    # img = np.dstack((curl, np.ones_like(curl), density))
    img = (np.clip(img, 0, 1) * 255).astype('uint8')
    img = Image.fromarray(img, mode='HSV').convert('RGB')
    if save_path is not None:
        img.save(save_path)
    return img


def save_figure(fig, save_path, close=True, axis_off=False):
    if axis_off:
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
    plt.savefig(save_path, bbox_inches='tight')
    if close:
        plt.close(fig)


def frames2gif(src_dir, save_path, fps=30):
    print("Convert frames to gif...")
    filenames = sorted([x for x in os.listdir(src_dir) if x.endswith('.png')])
    img_list = [Image.open(os.path.join(src_dir, name)) for name in filenames]
    img = img_list[0]
    img.save(fp=save_path, append_images=img_list[1:],
            save_all=True, duration=1 / fps * 1000, loop=0)
    print("Done.")
