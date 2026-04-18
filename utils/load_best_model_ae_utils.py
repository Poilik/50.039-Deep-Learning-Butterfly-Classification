import numpy as np
import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassFBetaScore


def _get_class_to_idx(dataset):
    if hasattr(dataset, "class_to_idx"):
        return dataset.class_to_idx
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "class_to_idx"):
        return dataset.dataset.class_to_idx
    return {"0_non-hybrid": 0, "1_hybrid": 1}


def _reconstruction_errors_and_labels(model, dataset, device, batch_size=32, num_workers=2):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    errors = []
    labels = []

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            x_hat = model(x)
            err = ((x_hat - x) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            errors.extend(err.tolist())
            labels.extend(y.numpy().tolist())

    return np.array(errors), np.array(labels)


def _f2_score_binary_torch(y_true, y_pred, device, positive_class=1, beta=2.0):
    metric = MulticlassFBetaScore(num_classes=2, average=None, beta=beta).to(device)
    metric.reset()
    y_true_t = torch.as_tensor(y_true, dtype=torch.long, device=device)
    y_pred_t = torch.as_tensor(y_pred, dtype=torch.long, device=device)
    metric.update(y_pred_t, y_true_t)
    return float(metric.compute().cpu().detach().numpy()[positive_class])


def _f2_scores_torch(y_true, y_pred, device, beta=2.0):
    macro_metric = MulticlassFBetaScore(num_classes=2, average="macro", beta=beta).to(device)
    per_class_metric = MulticlassFBetaScore(num_classes=2, average=None, beta=beta).to(device)

    y_true_t = torch.as_tensor(y_true, dtype=torch.long, device=device)
    y_pred_t = torch.as_tensor(y_pred, dtype=torch.long, device=device)

    macro_metric.reset()
    per_class_metric.reset()
    macro_metric.update(y_pred_t, y_true_t)
    per_class_metric.update(y_pred_t, y_true_t)

    f2_macro = float(macro_metric.compute().item())
    f2_per_class = per_class_metric.compute().cpu().detach().numpy()
    return f2_macro, f2_per_class


def load_best_model_ae(
    best_model,
    best,
    val_dataset,
    test_dataset,
    device,
    batch_size=32,
    num_workers=2,
    beta=2.0,
):
    state_dict = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in best["model_state_dict"].items()
    }
    best_model.load_state_dict(state_dict)
    best_model.eval()

    class_to_idx = _get_class_to_idx(val_dataset)
    non_hybrid_idx = class_to_idx.get("0_non-hybrid", 0)
    hybrid_idx = class_to_idx.get("1_hybrid", 1)

    val_err, val_y = _reconstruction_errors_and_labels(
        model=best_model,
        dataset=val_dataset,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )

    thresholds = np.linspace(val_err.min(), val_err.max(), 300)
    best_t = thresholds[0] if thresholds.size else 0.0
    best_val_f2 = -1.0

    for t in thresholds:
        val_pred = np.where(val_err > t, hybrid_idx, non_hybrid_idx)
        f2 = _f2_score_binary_torch(
            val_y,
            val_pred,
            device=device,
            positive_class=hybrid_idx,
            beta=beta,
        )
        if f2 > best_val_f2:
            best_val_f2 = f2
            best_t = t

    test_err, test_y = _reconstruction_errors_and_labels(
        model=best_model,
        dataset=test_dataset,
        batch_size=batch_size,
        device=device,
        num_workers=num_workers,
    )
    test_pred = np.where(test_err > best_t, hybrid_idx, non_hybrid_idx)
    test_loss = float(test_err.mean())
    test_f2_macro, test_f2_per_class = _f2_scores_torch(
        test_y,
        test_pred,
        device=device,
        beta=beta,
    )

    print("Test loss:", test_loss)
    print("Test F2 macro:", test_f2_macro)
    print("Test F2 per class:", test_f2_per_class)
    print(f"Test F2 for class 1 ({test_dataset.classes[1]}):", float(test_f2_per_class[1]))

    return {
        "threshold": float(best_t),
        "val_f2": float(best_val_f2),
        "test_loss": test_loss,
        "test_f2_macro": test_f2_macro,
        "test_f2_per_class": test_f2_per_class,
        "test_f2": float(test_f2_per_class[hybrid_idx]),
    }
