import torch.nn as nn
import pickle
import os

class AE(nn.Module):
    def __init__(self, latent_channels=64, dropout_p=0.1, noise_std=0.02):
        super().__init__()
        self.noise_std = noise_std

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, latent_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=dropout_p),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            noise = self.noise_std * nn.init.normal_(x.new_empty(x.size()))
            x = (x + noise).clamp(0.0, 1.0)

        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
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