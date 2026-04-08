from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

#roots are for image folder, sets are for ready datasets, can choose either depending on availability
def dataloader(train_root=None, val_root=None, test_root=None, train_set=None, val_set=None, test_set=None, transform_train=None, transform_eval=None, transform_val=None, transform_test=None, batch_size=32, num_workers=4):

    if train_root is not None and train_set is None:
        train_dataset = ImageFolder(root=train_root, transform=transform_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)   
        print("Classes:", train_dataset.classes)
        print("Class->idx:", train_dataset.class_to_idx)
        print("Total:", len(train_dataset))
    elif train_set is not None and train_root is None:
        train_dataset = train_set
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    else:
        train_dataset = None
        train_loader = None

    if val_root is not None and val_set is None:
        val_dataset = ImageFolder(root=val_root, transform=transform_eval)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print("Classes:", val_dataset.classes)
        print("Class->idx:", val_dataset.class_to_idx)
        print("Total:", len(val_dataset))
    elif val_set is not None and val_root is None:
        val_dataset = val_set
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        val_dataset = None
        val_loader = None

    if test_root is not None and test_set is None:
        test_dataset = ImageFolder(root=test_root, transform=transform_eval)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        print("Classes:", test_dataset.classes)
        print("Class->idx:", test_dataset.class_to_idx)
        print("Total:", len(test_dataset))
    elif test_set is not None and test_root is None:
        test_dataset = test_set
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        test_dataset = None
        test_loader = None

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader