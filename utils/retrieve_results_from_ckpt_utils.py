def results_from_ckpt(model, ckpt_file):
    history = model.load_checkpoint_history(ckpt_file)
    metrics = history[-1]["metrics"]
    return (
    metrics["train_loss_history"],
    metrics["val_loss_history"],
    metrics["train_f1_macro_history"],
    metrics["val_f1_macro_history"],
    metrics["train_f1_per_class_history"],
    metrics["val_f1_per_class_history"],
    metrics["train_f2_macro_history"],
    metrics["val_f2_macro_history"],
    metrics["train_f2_per_class_history"],
    metrics["val_f2_per_class_history"],
)