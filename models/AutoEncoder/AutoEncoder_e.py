import torch
import torch.nn as nn
import pickle
import os


LATENT_SWEEP_CHANNELS = (16, 32, 64)


def build_latent_sweep_models(device=None, latent_channels_values=LATENT_SWEEP_CHANNELS, **ae_kwargs):
    """Build AE models for a latent-channel sweep using the same architecture.

    Returns a dict keyed by latent size, e.g. "ae_latent_16".
    """
    models = {}
    for c in latent_channels_values:
        model = AE(latent_channels=c, **ae_kwargs)
        if device is not None:
            model = model.to(device)
        models[f"ae_latent_{c}"] = model
    return models

class AE(nn.Module):
    def __init__(self, latent_channels=64, dropout_p=0.1, noise_std=0.02):
        super().__init__()
        self.noise_std = noise_std
        gn_groups = 8

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.GroupNorm(gn_groups, 32),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.GroupNorm(gn_groups, 64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.GroupNorm(gn_groups, 128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.enc4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.GroupNorm(gn_groups, 256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.to_latent = nn.Sequential(
            nn.Conv2d(256, latent_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout2d(p=dropout_p),
        )

        self.from_latent = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 256, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.fuse4 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(gn_groups, 256),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.GroupNorm(gn_groups, 128),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.fuse3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(gn_groups, 128),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.GroupNorm(gn_groups, 64),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.fuse2 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(gn_groups, 64),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.GroupNorm(gn_groups, 32),
            nn.LeakyReLU(0.1, inplace=True),
        )

        self.fuse1 = nn.Sequential(
            nn.Conv2d(32 + 32, 32, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(gn_groups, 32),
            nn.LeakyReLU(0.1, inplace=True),
        )
        self.out_head = nn.Sequential(
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        if self.training and self.noise_std > 0:
            noise = self.noise_std * nn.init.normal_(x.new_empty(x.size()))
            x = (x + noise).clamp(0.0, 1.0)

        e1 = self.enc1(x)   # 64 -> 32
        e2 = self.enc2(e1)  # 32 -> 16
        e3 = self.enc3(e2)  # 16 -> 8
        e4 = self.enc4(e3)  # 8 -> 4

        z = self.to_latent(e4)
        d4 = self.from_latent(z)
        d4 = self.fuse4(torch.cat([d4, e4], dim=1))

        d3 = self.up1(d4)
        d3 = self.fuse3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = self.fuse2(torch.cat([d2, e2], dim=1))

        d1 = self.up3(d2)
        d1 = self.fuse1(torch.cat([d1, e1], dim=1))

        x_hat = self.out_head(d1)
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