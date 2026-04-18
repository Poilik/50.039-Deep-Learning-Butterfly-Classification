import numpy as np
import torch as torch
import torch.nn as nn
import pickle
import os


class VisionTransformerProcessor:
    def __init__(self, img_size, patch_size, embed_dim, device):
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size//patch_size)**2
        self.embed_dim = embed_dim
        self.grid_size = (img_size//patch_size, img_size//patch_size)
        self.device = device
        self.linear_proj = nn.Conv2d(in_channels = 3, out_channels = embed_dim, 
                                     kernel_size = patch_size, stride = patch_size).to(device)
        self.positional_encodings = torch.tensor(self.generate_2d_positional_encoding(),dtype=torch.float32).to(device)
    
    def generate_2d_positional_encoding(self):
        num_rows, num_cols = self.grid_size
        row_pos = np.arange(num_rows).reshape(-1, 1)
        col_pos = np.arange(num_cols).reshape(-1, 1)
        d = np.arange(self.embed_dim//2).reshape(1, -1)

        angle_rates = 1/np.power(10000, (2*d)/self.embed_dim)
        row_encoding = np.concatenate([np.sin(row_pos*angle_rates), np.cos(row_pos*angle_rates)], axis = -1)
        col_encoding = np.concatenate([np.sin(col_pos*angle_rates), np.cos(col_pos*angle_rates)], axis = -1)

        row_encoding = np.tile(row_encoding[:, np.newaxis, :], (1, num_cols, 1))
        col_encoding = np.tile(col_encoding[np.newaxis, :, :], (num_rows, 1, 1))
        pos_encoding = row_encoding + col_encoding
        return pos_encoding.reshape(-1, self.embed_dim)

    def process_images(self, images):
        patches = self.linear_proj(images)
        patches = patches.flatten(2).transpose(1, 2) 
        patches_with_pos = patches + self.positional_encodings.unsqueeze(0) 
        return patches_with_pos

class ViT(nn.Module):
    def __init__(self, embed_dim, num_patches, num_classes, num_heads, num_layers, mlp_dim, dropout):
        super(ViT, self).__init__()
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model = embed_dim,
                                                                        nhead = num_heads,
                                                                        dim_feedforward = mlp_dim,
                                                                        dropout = dropout,
                                                                        activation = 'gelu',
                                                                        batch_first = True),
                                            num_layers=num_layers)
        self.mlp_head = nn.Sequential(nn.LayerNorm(embed_dim),
                                      nn.Linear(embed_dim, num_classes))

    def forward(self, x):
        batch_size, num_patches, embed_dim = x.shape
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim = 1)
        x = self.encoder(x)
        cls_output = x[:, 0]
        logits = self.mlp_head(cls_output)
        return logits

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