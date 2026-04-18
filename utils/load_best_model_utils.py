import torch
from torch import nn
from utils.test_utils import evaluate_split

def load_best_model(best_model, best, test_dataset, device):

    state_dict = {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in best["model_state_dict"].items()
    }
    best_model.load_state_dict(state_dict)
    best_model.eval()

    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_split(
        model=best_model,
        dataset=test_dataset,
        batch_size=32,
        criterion=criterion,
        device=device,
        num_workers=0
    )

    print("Test loss:", test_metrics["loss"])
    print("Test F2 macro:", test_metrics["f2_macro"])
    print("Test F2 per class:", test_metrics["f2_per_class"])
    print(f"Test F2 for class 1 ({test_dataset.classes[1]}):", float(test_metrics["f2_per_class"][1]))