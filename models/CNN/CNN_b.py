import torch.nn as nn
import pickle
import os

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

def conv4x4(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=4,
                     stride=stride, padding=1, bias=False)

def conv1x1(in_channels, out_channels, stride=2):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)

class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.conv1 = conv4x4(16, 32)
        self.conv2 = conv4x4(32, 64)
        self.conv3 = conv1x1(64, 128)
        self.dp = nn.Dropout(0.5)
        self.bn = nn.BatchNorm2d(16)
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        out = self.conv(x)
        out = self.dp(out)
        out = self.bn(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.dp(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.dp(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.dp(out)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def save_checkpoint_append(
            self,
            ckpt_file,
            optimizer=None,
            epoch=None,
            train_loss_history=None,
            val_loss_history=None,
            train_f1_macro_history=None,
            val_f1_macro_history=None,
            train_f1_per_class_history=None,
            val_f1_per_class_history=None,
            train_f2_macro_history=None,
            val_f2_macro_history=None,
            train_f2_per_class_history=None,
            val_f2_per_class_history=None,
            ):
            os.makedirs(os.path.dirname(ckpt_file), exist_ok=True)

            record = {
                "epoch": epoch,
                "model_state_dict": self.state_dict(),
                "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
                "metrics": {
                    "train_loss_history": train_loss_history,
                    "val_loss_history": val_loss_history,
                    "train_f1_macro_history": train_f1_macro_history,
                    "val_f1_macro_history": val_f1_macro_history,
                    "train_f1_per_class_history": train_f1_per_class_history.tolist() if hasattr(train_f1_per_class_history, "tolist") else train_f1_per_class_history,
                    "val_f1_per_class_history": val_f1_per_class_history.tolist() if hasattr(val_f1_per_class_history, "tolist") else val_f1_per_class_history,
                    "train_f2_macro_history": train_f2_macro_history,
                    "val_f2_macro_history": val_f2_macro_history,
                    "train_f2_per_class_history": train_f2_per_class_history.tolist() if hasattr(train_f2_per_class_history, "tolist") else train_f2_per_class_history,
                    "val_f2_per_class_history": val_f2_per_class_history.tolist() if hasattr(val_f2_per_class_history, "tolist") else val_f2_per_class_history
                }
            }

            with open(ckpt_file, "ab") as f:
                pickle.dump(record, f)

    def load_checkpoint_history(self, ckpt_file):
        history = []
        with open(ckpt_file, "rb") as f:
            while True:
                try:
                    history.append(pickle.load(f))
                except EOFError:
                    break
        return history