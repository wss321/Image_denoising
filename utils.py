import matplotlib.pyplot as plt
import numpy as np


def visualize_image(image, visual=True, title=''):
    """Displays the image in a zjl-230 matrix"""
    num_dims = len(list(np.shape(image)))
    if visual:
        plt.figure()
        plt.title(title)
        if num_dims == 2:
            plt.imshow(image, cmap='gray')  # , interpolation='nearest'
        elif num_dims == 3:
            plt.imshow(image)
        plt.show()
    return image
