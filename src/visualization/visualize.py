## Plot 5 random images histograms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

def plot_images_histogram(df,n_samples,key,bins=128):
    random_imgs = [np.asarray(Image.open(df.iloc[idx][key])) for idx in np.random.randint(0,len(df),size=n_samples)]
    rows = n_samples
    cols = 2

    gs = gridspec.GridSpec(rows, cols*2, wspace=0.5, hspace=0.1)
    fig = plt.figure(figsize=(10,rows*6))
    for i, image in enumerate(random_imgs):

        ax_image = plt.subplot(gs[i ,0])
        ax_image.imshow(image)
        ax_image.set_aspect('auto')
        ax_image.axis('off')

        # Histogram subplot
        ax_hist = plt.subplot(gs[i,1])
        ax_hist.hist(image.flatten(), bins=bins)

    plt.show()

def plot_norm_images_histogram(df,n_samples,key,mean,std,bins=128):
    random_imgs = [(np.asarray(Image.open(df.iloc[idx][key])) -mean) / std for idx in np.random.randint(0,len(df),size=n_samples)]
    rows = n_samples
    cols = 2

    gs = gridspec.GridSpec(rows, cols*2, wspace=0.5, hspace=0.1)
    fig = plt.figure(figsize=(10,rows*6))
    for i, image in enumerate(random_imgs):
        ax_image = plt.subplot(gs[i ,0])
        ax_image.imshow(image)
        ax_image.set_aspect('auto')
        ax_image.axis('off')

        # Histogram subplot
        ax_hist = plt.subplot(gs[i,1])
        ax_hist.hist(image.flatten(), bins=bins)

    plt.show()