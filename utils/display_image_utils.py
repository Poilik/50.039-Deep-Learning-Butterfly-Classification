import matplotlib.pyplot as plt

# Display original vs augmented samples
def display_original_vs_augmented(train_dataset, train_dataset_aug):
    class_names = train_dataset_aug.classes
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    for i, ax in enumerate(axes[0]):
        img, label = train_dataset[i]  # original transform view
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.set_title(f"Original: {class_names[label]}")
        ax.axis("off")

    for i, ax in enumerate(axes[1]):
        img, label = train_dataset_aug[i]  # augmented transform view
        img_np = img.permute(1, 2, 0).numpy().clip(0, 1)
        ax.imshow(img_np)
        ax.set_title(f"Augmented: {class_names[label]}")
        ax.axis("off")

    plt.suptitle("Original vs Augmented Images", fontsize=16)
    plt.tight_layout()
    plt.show()