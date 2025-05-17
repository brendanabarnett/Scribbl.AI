import matplotlib.pyplot as plt

def display_word_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(len(images) * 2, 2))
    for ax, img in zip(axes, images):
        ax.imshow(img.squeeze(), cmap='gray')
        ax.axis('off')
    plt.show()