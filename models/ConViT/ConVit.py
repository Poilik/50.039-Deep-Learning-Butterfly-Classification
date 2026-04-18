import numpy as np
import torch as torch
import torch.nn as nn
import pickle
import os

class ConvViTProcessor:
    def __init__(self, img_size, patch_size, embed_dim, padding, device):
        self.img_size = img_size
        self.patch_size = patch_size
        self.padding = padding
        self.embed_dim = embed_dim
        self.device = device
        self.conv_emb = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels = embed_dim // 2, 
                                     kernel_size = patch_size, padding = self.padding),
            nn.BatchNorm2d(embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = embed_dim // 2, out_channels = embed_dim, 
                                     kernel_size = patch_size, padding = self.padding),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, kernel_size=patch_size, stride=patch_size)        
).to(device)
        self.positional_encodings = None

    def process_images(self, images):
        patches = self.conv_emb(images) 
        B, C, H, W = patches.size()
        self.positional_encodings = nn.Parameter(torch.zeros(1, H*W, embed_dim)).to(device)
        patches = patches.flatten(2).transpose(1, 2)  
        patches_with_pos = patches + self.positional_encodings
        return patches_with_pos

class ViT(nn.Module):
    def __init__(self, embed_dim, num_patches, num_classes, num_heads, num_layers, mlp_dim, dropout):
        super(ViT, self).__init__()
        self.num_patches = num_patches
        self.cls_token = nn.Parameter(torch.randint(low=0, high=5, size=(1, 1, embed_dim)).to(torch.float32))
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