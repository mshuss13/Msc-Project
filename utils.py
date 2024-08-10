from torchvision import datasets, transforms
import torch

def get_data_loaders(batch_size, train_dir='data/Training', test_dir='data/Testing', img_size=(150, 150)):
    data_transforms = transforms.Compose([
        transforms.Resize(img_size),  # Resize images to a fixed size
        transforms.ToTensor()
    ])

    train_dataset = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=data_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_dataset.classes
