import torch
from torchmetrics.classification import MulticlassF1Score
from torchmetrics.classification import MulticlassFBetaScore
from torch.utils.data import DataLoader

#evaluate f1 and f2 score for test set
def evaluate_split(model, dataset, batch_size, criterion, device, num_workers=4, embeddings = None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    f1_macro = MulticlassF1Score(num_classes=2, average="macro").to(device)
    f1_per_class = MulticlassF1Score(num_classes=2, average=None).to(device)
    f2_macro = MulticlassFBetaScore(num_classes=2, average="macro", beta=2.0).to(device)
    f2_per_class = MulticlassFBetaScore(num_classes=2, average=None, beta=2.0).to(device)

    model.eval()
    total_loss = 0.0
    f1_macro.reset()
    f1_per_class.reset()
    f2_macro.reset()
    f2_per_class.reset()

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            if (embeddings):
                images = embeddings(images)
            outputs = model(images)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            preds = outputs.argmax(dim=1)
            f1_macro.update(preds, labels)
            f1_per_class.update(preds, labels)
            f2_macro.update(preds, labels)
            f2_per_class.update(preds, labels)

    return {
        "loss": total_loss / len(loader),
        "f1_macro": f1_macro.compute().item(),
        "f1_per_class": f1_per_class.compute().cpu().numpy(),
        "f2_macro": f2_macro.compute().item(),
        "f2_per_class": f2_per_class.compute().cpu().numpy()
    }