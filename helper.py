'''
Helper and zip functions.
'''

import os, torch
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F


def show_image(img: torch.Tensor, delay=1000):
    """Shows an image.
    """
    plt.imshow(F.to_pil_image(img))
    plt.show()
    plt.pause(delay / 1000)