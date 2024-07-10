import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torchvision


def img_save(img, save_path, iteration, prefix):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # Make a grid of images
    # npimg = torchvision.utils.make_grid(img, value_range=(-1, 1), padding=0, nrow=20)
    # Move the grid to the CPU and convert it to a NumPy array
    npimg = img.cpu().numpy()
    # Unnormalize the image
    npimg = npimg / 2 + 0.5
    # Transpose the image to get it in the right format for displaying
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis("off")
    save_path = os.path.join(save_path, f"{prefix}_{iteration}.png")
    plt.savefig(save_path)