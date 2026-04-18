import numpy as np
import matplotlib.pyplot as plt

#Plotting train and val values on same graphs, one each for loss, f1 score and f2 score
def plot_training_curves(model_name, results, class_idx=1, class_name="1_hybrid"):
    """
    Expects results from train_and_evaluate in this order:
    (
        train_loss_list,
        val_loss_list,
        train_f1_macro_list,
        val_f1_macro_list,
        train_f1_per_class_list,
        val_f1_per_class_list,
        train_f2_macro_list,
        val_f2_macro_list,
        train_f2_per_class_list,
        val_f2_per_class_list
    )
    """
    (
        train_loss, val_loss,
        _, _,
        _, _,
        _, _,
        train_f2_per_class, val_f2_per_class
    ) = results

    train_f2_c = [float(np.array(x)[class_idx]) for x in train_f2_per_class]
    val_f2_c   = [float(np.array(x)[class_idx]) for x in val_f2_per_class]

    epochs = range(1, len(train_loss) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(18, 5))
    fig.suptitle(model_name, fontsize=14)

    axes[0].plot(epochs, train_loss, label="Train Loss")
    axes[0].plot(epochs, val_loss, label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_f2_c, label=f"Train F2 ({class_name})")
    axes[1].plot(epochs, val_f2_c, label=f"Val F2 ({class_name})")
    axes[1].set_title(f"F2 Class {class_idx}: {class_name}")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("F2")
    axes[1].set_ylim(0, 1)
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    # Force immediate rendering per model in notebooks so label and graph appear together.
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()