from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def load_train_val_data(batch_size=64):
    
    train_dir = "replace_with_your_train_directory_path"
    val_dir = "replace_with_your_val_directory_path"
    
    transforms_all = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=transforms_all)
    val_dataset = datasets.ImageFolder(val_dir, transform=transforms_all)
    

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True,

        )    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
        persistent_workers=True
    )


    return train_loader, val_loader