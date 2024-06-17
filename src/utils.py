import os
from typing import Tuple
from matplotlib import pyplot as plt

def find_files(folder, ext: Tuple):
    out_files = []

    # Walk through the directory and its subdirectories
    for root, dirs, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(ext):
                out_files.append(os.path.join(root, file))

    print(f'Found {len(out_files)} image files in {folder}')
    return out_files

def num_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)

def show_image(x, idx, img_size, figure=True):
    if figure:
      fig = plt.figure()

    x = x.view(-1, *img_size, 3)
    plt.imshow(x[idx].cpu().numpy())

def show_comparison(x, x_hat, idx):
    fig = plt.figure()
    plt.subplot(1, 2, 1)
    show_image(x, idx, False)
    plt.title("Original")
    plt.subplot(1, 2, 2)
    show_image(x_hat, idx, False)
    plt.title("Reconstruction")
