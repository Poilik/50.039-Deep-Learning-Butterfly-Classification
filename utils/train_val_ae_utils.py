import copy
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassFBetaScore


def _get_class_to_idx(dataset):
    if hasattr(dataset, "class_to_idx"):
        return dataset.class_to_idx
    if hasattr(dataset, "dataset") and hasattr(dataset.dataset, "class_to_idx"):
        return dataset.dataset.class_to_idx
    return {"0_non-hybrid": 0, "1_hybrid": 1}


def _reconstruction_errors_and_labels(model, loader, device):
    errors = []
    labels = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            x_hat = model(x)
            err = ((x_hat - x) ** 2).mean(dim=(1, 2, 3)).cpu().numpy()
            errors.extend(err.tolist())
            labels.extend(y.numpy().tolist())

    return np.array(errors), np.array(labels)


def _weighted_reconstruction_loss(x_hat, x, l1_weight=0.7, mse_weight=0.3):
    return l1_weight * F.l1_loss(x_hat, x) + mse_weight * F.mse_loss(x_hat, x)

def train_and_evaluate_ae(
    model,
    train_loader,
    val_loader,
    optimizer,
    num_epochs,
    ckpt_file,
    device,
    criterion=None,
    patience=6,
    min_delta=1e-3,
    early_stop_metric="val_f2_class1",
    restore_best_weights=True,
    use_weighted_recon_loss=True,
    l1_weight=0.7,
    mse_weight=0.3,
):
    train_loss_list = []
    val_loss_list = []
    train_f1_macro_list = []
    val_f1_macro_list = []
    train_f1_per_class_list = []
    val_f1_per_class_list = []
    train_f2_macro_list = []
    val_f2_macro_list = []
    train_f2_per_class_list = []
    val_f2_per_class_list = []

    if not use_weighted_recon_loss and criterion is None:
        raise ValueError("criterion must be provided when use_weighted_recon_loss=False")

    class_to_idx = _get_class_to_idx(val_loader.dataset)
    non_hybrid_idx = class_to_idx.get("0_non-hybrid", 0)
    hybrid_idx = class_to_idx.get("1_hybrid", 1)

    train_f1_macro_metric = MulticlassF1Score(num_classes=2, average="macro").to(device)
    train_f1_per_class_metric = MulticlassF1Score(num_classes=2, average=None).to(device)
    val_f1_macro_metric = MulticlassF1Score(num_classes=2, average="macro").to(device)
    val_f1_per_class_metric = MulticlassF1Score(num_classes=2, average=None).to(device)

    train_f2_macro_metric = MulticlassFBetaScore(num_classes=2, average="macro", beta=2.0).to(device)
    train_f2_per_class_metric = MulticlassFBetaScore(num_classes=2, average=None, beta=2.0).to(device)
    val_f2_macro_metric = MulticlassFBetaScore(num_classes=2, average="macro", beta=2.0).to(device)
    val_f2_per_class_metric = MulticlassFBetaScore(num_classes=2, average=None, beta=2.0).to(device)

    best_score = float("-inf")
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        running_train_loss = 0.0

        for x, _ in train_loader:
            x = x.to(device, non_blocking=True)

            x_hat = model(x)
            if use_weighted_recon_loss:
                loss = _weighted_reconstruction_loss(
                    x_hat,
                    x,
                    l1_weight=l1_weight,
                    mse_weight=mse_weight,
                )
            else:
                loss = criterion(x_hat, x)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item()

        avg_train_loss = running_train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        model.eval()
        running_val_loss = 0.0

        with torch.no_grad():
            for x, _ in val_loader:
                x = x.to(device, non_blocking=True)
                x_hat = model(x)
                if use_weighted_recon_loss:
                    loss = _weighted_reconstruction_loss(
                        x_hat,
                        x,
                        l1_weight=l1_weight,
                        mse_weight=mse_weight,
                    )
                else:
                    loss = criterion(x_hat, x)
                running_val_loss += loss.item()

        avg_val_loss = running_val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)

        train_err, train_y = _reconstruction_errors_and_labels(model, train_loader, device)
        val_err, val_y = _reconstruction_errors_and_labels(model, val_loader, device)

        thresholds = np.linspace(val_err.min(), val_err.max(), 300)
        if thresholds.size == 0:
            thresholds = np.array([0.0], dtype=np.float32)

        best_t = thresholds[0]
        best_val_f2_class1 = -1.0

        for t in thresholds:
            val_pred_tmp = np.where(val_err > t, hybrid_idx, non_hybrid_idx)
            val_f2_per_class_metric.reset()
            val_pred_tmp_t = torch.as_tensor(val_pred_tmp, dtype=torch.long, device=device)
            val_y_t = torch.as_tensor(val_y, dtype=torch.long, device=device)
            val_f2_per_class_metric.update(val_pred_tmp_t, val_y_t)
            current_val_f2_class1 = float(val_f2_per_class_metric.compute().cpu().detach().numpy()[1])
            if current_val_f2_class1 > best_val_f2_class1:
                best_val_f2_class1 = current_val_f2_class1
                best_t = t

        train_pred = np.where(train_err > best_t, hybrid_idx, non_hybrid_idx)
        val_pred = np.where(val_err > best_t, hybrid_idx, non_hybrid_idx)

        train_pred_t = torch.as_tensor(train_pred, dtype=torch.long, device=device)
        train_y_t = torch.as_tensor(train_y, dtype=torch.long, device=device)
        val_pred_t = torch.as_tensor(val_pred, dtype=torch.long, device=device)
        val_y_t = torch.as_tensor(val_y, dtype=torch.long, device=device)

        train_f1_macro_metric.reset()
        train_f1_per_class_metric.reset()
        train_f2_macro_metric.reset()
        train_f2_per_class_metric.reset()
        val_f1_macro_metric.reset()
        val_f1_per_class_metric.reset()
        val_f2_macro_metric.reset()
        val_f2_per_class_metric.reset()

        train_f1_macro_metric.update(train_pred_t, train_y_t)
        train_f1_per_class_metric.update(train_pred_t, train_y_t)
        train_f2_macro_metric.update(train_pred_t, train_y_t)
        train_f2_per_class_metric.update(train_pred_t, train_y_t)

        val_f1_macro_metric.update(val_pred_t, val_y_t)
        val_f1_per_class_metric.update(val_pred_t, val_y_t)
        val_f2_macro_metric.update(val_pred_t, val_y_t)
        val_f2_per_class_metric.update(val_pred_t, val_y_t)

        train_f1_macro_list.append(train_f1_macro_metric.compute().item())
        train_f1_per_class_list.append(train_f1_per_class_metric.compute().cpu().detach().numpy())
        train_f2_macro_list.append(train_f2_macro_metric.compute().item())
        train_f2_per_class_list.append(train_f2_per_class_metric.compute().cpu().detach().numpy())

        val_f1_macro_list.append(val_f1_macro_metric.compute().item())
        val_f1_per_class_list.append(val_f1_per_class_metric.compute().cpu().detach().numpy())
        val_f2_macro_list.append(val_f2_macro_metric.compute().item())
        val_f2_per_class_list.append(val_f2_per_class_metric.compute().cpu().detach().numpy())

        model.save_checkpoint_append(
            ckpt_file=ckpt_file,
            optimizer=optimizer,
            epoch=epoch + 1,
            train_loss_history=train_loss_list,
            val_loss_history=val_loss_list,
            train_f1_macro_history=train_f1_macro_list,
            val_f1_macro_history=val_f1_macro_list,
            train_f1_per_class_history=train_f1_per_class_list,
            val_f1_per_class_history=val_f1_per_class_list,
            train_f2_macro_history=train_f2_macro_list,
            val_f2_macro_history=val_f2_macro_list,
            train_f2_per_class_history=train_f2_per_class_list,
            val_f2_per_class_history=val_f2_per_class_list,
        )

        if early_stop_metric == "val_f2_class1":
            current_score = val_f2_per_class_list[-1][1]
        elif early_stop_metric == "val_f2_macro":
            current_score = val_f2_macro_list[-1]
        elif early_stop_metric == "val_loss":
            current_score = -avg_val_loss
        else:
            current_score = val_f2_per_class_list[-1][1]

        if current_score > best_score + min_delta:
            best_score = current_score
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  [Epoch {epoch + 1}] Improvement! New best score: {current_score:.6f}")
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 5 == 0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Train F1-Macro: {train_f1_macro_list[-1]:.4f}, "
                f"Val F1-Macro: {val_f1_macro_list[-1]:.4f}, "
                f"Train F1-Per-Class: {train_f1_per_class_list[-1]}, "
                f"Val F1-Per-Class: {val_f1_per_class_list[-1]}, "
                f"Train F2-Macro: {train_f2_macro_list[-1]:.4f}, "
                f"Val F2-Macro: {val_f2_macro_list[-1]:.4f}, "
                f"Train F2-Per-Class: {train_f2_per_class_list[-1]}, "
                f"Val F2-Per-Class: {val_f2_per_class_list[-1]}, "
                f"No improvement: {epochs_no_improve}/{patience}"
            )

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch + 1}. No improvement for {patience} epochs.")
            break

    if restore_best_weights and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored model weights from best epoch (score: {best_score:.6f})")

    return (
        train_loss_list,
        val_loss_list,
        train_f1_macro_list,
        val_f1_macro_list,
        train_f1_per_class_list,
        val_f1_per_class_list,
        train_f2_macro_list,
        val_f2_macro_list,
        train_f2_per_class_list,
        val_f2_per_class_list,
    )