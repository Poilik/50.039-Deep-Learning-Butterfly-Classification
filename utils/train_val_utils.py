import copy
from math import inf
import torch
import torch.nn as nn
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassFBetaScore
from utils.dataloader_utils import dataloader

#train using weighted cross entropy loss but report balanced cross entropy loss, 
# save history of train and val loss, f1 macro, f1 per class, f2 macro and f2 per class for each epoch, 
# also implement early stopping based on val_f2_class1 with patience of 6 epochs and min_delta of 0.001, 
# save best model checkpoint based on val_f2_class1, also save checkpoint every 5 epochs with all history so far

def train_and_evaluate(
    model, 
    train_set, 
    val_set, 
    optimizer, 
    num_epochs, 
    batch_size, 
    class_weights_val, 
    ckpt_file,
    device,
    num_workers=4, 
    patience=6,
    min_delta=1e-3,
    early_stop_metric="val_f2_class1",
    restore_best_weights=True):

    _, _, _, train_loader, val_loader, _ = dataloader(
        train_set=train_set,
        val_set=val_set,
        batch_size=batch_size,
        num_workers=num_workers
    )

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
    
    hybrid_idx = train_set.class_to_idx["1_hybrid"]
    class_weights = torch.ones(len(train_set.classes), device=device)
    class_weights[hybrid_idx] = class_weights_val

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train_f1_macro_metric = MulticlassF1Score(num_classes=2, average="macro").to(device)
    train_f1_per_class_metric = MulticlassF1Score(num_classes=2, average=None).to(device)
    val_f1_macro_metric = MulticlassF1Score(num_classes=2, average="macro").to(device)
    val_f1_per_class_metric = MulticlassF1Score(num_classes=2, average=None).to(device)

    train_f2_macro_metric = MulticlassFBetaScore(num_classes=2, average="macro", beta=2.0).to(device)
    train_f2_per_class_metric = MulticlassFBetaScore(num_classes=2, average=None, beta=2.0).to(device)
    val_f2_macro_metric = MulticlassFBetaScore(num_classes=2, average="macro", beta=2.0).to(device)
    val_f2_per_class_metric = MulticlassFBetaScore(num_classes=2, average=None, beta=2.0).to(device)

    best_score = -inf
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward() 
            optimizer.step()

            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_loss_list.append(avg_train_loss)

        model.eval()
        train_f1_macro_metric.reset()
        train_f1_per_class_metric.reset()
        train_f2_macro_metric.reset()
        train_f2_per_class_metric.reset()

        with torch.no_grad():
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                preds = outputs.argmax(dim=1)

                train_f1_macro_metric.update(preds, labels)
                train_f1_per_class_metric.update(preds, labels)
                train_f2_macro_metric.update(preds, labels)
                train_f2_per_class_metric.update(preds, labels)

        train_f1_macro_list.append(train_f1_macro_metric.compute().item())
        train_f1_per_class_list.append(train_f1_per_class_metric.compute().cpu().detach().numpy())
        train_f2_macro_list.append(train_f2_macro_metric.compute().item())
        train_f2_per_class_list.append(train_f2_per_class_metric.compute().cpu().detach().numpy())

        epoch_val_loss = 0.0
        val_f1_macro_metric.reset()
        val_f1_per_class_metric.reset()
        val_f2_macro_metric.reset()
        val_f2_per_class_metric.reset()

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                epoch_val_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                val_f1_macro_metric.update(predicted, labels)
                val_f1_per_class_metric.update(predicted, labels)
                val_f2_macro_metric.update(predicted, labels)
                val_f2_per_class_metric.update(predicted, labels)

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_loss_list.append(avg_val_loss)
        val_f1_macro_list.append(val_f1_macro_metric.compute().item())
        val_f1_per_class_list.append(val_f1_per_class_metric.compute().cpu().detach().numpy())
        val_f2_macro_list.append(val_f2_macro_metric.compute().item())
        val_f2_per_class_list.append(val_f2_per_class_metric.compute().cpu().detach().numpy())

        #save checkpoint
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
            val_f2_per_class_history=val_f2_per_class_list
        )

        #early stopping based on f2 score
        if early_stop_metric == "val_f2_class1":
            current_score = val_f2_per_class_list[-1][1]  # F2 for class 1 (hybrid)
        elif early_stop_metric == "val_f2_macro":
            current_score = val_f2_macro_list[-1]
        elif early_stop_metric == "val_loss":
            current_score = -avg_val_loss  # negative so higher is better
        else:
            current_score = val_f2_per_class_list[-1][1]

        if current_score > best_score + min_delta:
            best_score = current_score
            epochs_no_improve = 0
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"  [Epoch {epoch+1}] Improvement! New best score: {current_score:.6f}")
        else:
            epochs_no_improve += 1

        #report every 5 epochs
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
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
                f"No improvement: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}. "
                  f"No improvement for {patience} epochs.")
            break

        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], "
                  f"Epoch [{epoch+1}/{num_epochs}], "
                f"Train Loss: {avg_train_loss:.4f}, "
                f"Val Loss: {avg_val_loss:.4f}, "
                f"Train F1-Macro: {train_f1_macro_list[-1]:.4f}, "
                f"Val F1-Macro: {val_f1_macro_list[-1]:.4f}, "
                f"Train F1-Per-Class: {train_f1_per_class_list[-1]}, "
                f"Val F1-Per-Class: {val_f1_per_class_list[-1]}",
                f"Train F2-Macro: {train_f2_macro_list[-1]:.4f}, "
                f"Val F2-Macro: {val_f2_macro_list[-1]:.4f}, "
                f"Train F2-Per-Class: {train_f2_per_class_list[-1]}, "
                f"Val F2-Per-Class: {val_f2_per_class_list[-1]}")
    
    # Load best model weights after training
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
        val_f2_per_class_list
    )