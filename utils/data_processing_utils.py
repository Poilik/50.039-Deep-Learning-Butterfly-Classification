import os
from PIL import Image
from pathlib import Path
import shutil
import random
import torchvision.transforms as transforms

def find_corrupted_images(root_dir):
    """Find all corrupted/truncated images in a directory tree"""
    corrupted = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                img_path = os.path.join(root, file)
                try:
                    img = Image.open(img_path)
                    img.load()
                except Exception as e:
                    corrupted.append({
                        'path': img_path,
                        'error': str(e)
                    })
    
    return corrupted

def split_dataset(
    input_dir,
    output_dir,
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42,
    dry_run=False
):
    random.seed(seed)

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    splits = ['train', 'val', 'test']

    for split in splits:
        for class_name in os.listdir(input_dir):
            class_path = os.path.join(output_dir, split, class_name)
            if not dry_run:
                os.makedirs(class_path, exist_ok=True)
            else:
                print(f"[DRY RUN] Would create: {class_path}")

    for class_name in os.listdir(input_dir):
        class_input_path = os.path.join(input_dir, class_name)

        if not os.path.isdir(class_input_path):
            continue

        files = [
            f for f in os.listdir(class_input_path)
            if os.path.isfile(os.path.join(class_input_path, f))
        ]

        random.shuffle(files)

        total = len(files)
        train_end = int(total * train_ratio)
        val_end = train_end + int(total * val_ratio)

        split_map = {
            'train': files[:train_end],
            'val': files[train_end:val_end],
            'test': files[val_end:]
        }

        for split, file_list in split_map.items():
            for file in file_list:
                src = os.path.join(class_input_path, file)
                dst = os.path.join(output_dir, split, class_name, file)

                if dry_run:
                    print(f"[DRY RUN] {src} -> {dst}")
                else:
                    shutil.copy2(src, dst)

    print("Dry run complete!" if dry_run else "Dataset split complete!")

def resize_images(src_root, dst_root, size=(224, 224)):
    """Resize all images in src_root and save to dst_root while preserving directory structure"""
    resize_tf = transforms.Resize(size)
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    count = 0
    for img_path in Path(src_root).rglob("*"):
        if img_path.is_file() and img_path.suffix.lower() in valid_ext:
            rel_path = img_path.relative_to(src_root)
            out_path = Path(dst_root) / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)

            with Image.open(img_path) as im:
                im = im.convert("RGB")
                im_resized = resize_tf(im)
                im_resized.save(out_path, quality=95)

            count += 1

    print(f"Done. Resized {count} images to: {dst_root}")