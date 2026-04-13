from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import CIFAR100, SVHN, SUN397, StanfordCars, Places365, Flowers102, LSUN
from torchvision import transforms

def get_traditional_dataset(dataset_name, batch_size=32, num_workers=2, transform_data=None, only_test=False):
    if transform_data is None:
        transform_data = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    if dataset_name.lower() == 'cifar100':
        test_dataset = CIFAR100(root='./data', train=False, download=True, transform=transform_data)
        if not only_test:
            train_dataset = CIFAR100(root='./data', train=True, download=True, transform=transform_data)
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    elif dataset_name.lower() == 'svhn':
        test_dataset = SVHN(root='./data', split='test', download=True, transform=transform_data)
        if not only_test:
            train_dataset = SVHN(root='./data', split='train', download=True, transform=transform_data)
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    elif dataset_name.lower() == 'sun397':
        # SUN397 does not have a split argument
        test_dataset = SUN397(root='./data', transform=transform_data, download=True)
        if not only_test:
            train_size = int(0.8 * len(test_dataset))
            val_size = len(test_dataset) - train_size
            train_dataset, val_dataset = random_split(test_dataset, [train_size, val_size])

    elif dataset_name.lower() == 'stanford_cars':
        test_dataset = StanfordCars(root='./data', split='test', download=True, transform=transform_data)
        if not only_test:
            train_dataset = StanfordCars(root='./data', split='train', download=True, transform=transform_data)
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    elif dataset_name.lower() == 'places365':
        test_dataset = Places365(root='./data', split='val', download=True, transform=transform_data)
        if not only_test:
            train_dataset = Places365(root='./data', split='train-standard', download=True, transform=transform_data)
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    elif dataset_name.lower() == 'oxford_flowers102':
        test_dataset = Flowers102(root='./data', split='test', download=True, transform=transform_data)
        if not only_test:
            train_dataset = Flowers102(root='./data', split='train', download=True, transform=transform_data)
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    elif dataset_name.lower() == 'lsun':
        # LSUN uses 'classes' flag for test/train splits and does not use download=True
        test_dataset = LSUN(root='./data', classes='test', transform=transform_data)
        if not only_test:
            train_dataset = LSUN(root='./data', classes='train', transform=transform_data)
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])

    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    if only_test:
        return test_dataset 
    return train_dataset, val_dataset, test_dataset

def get_traditional_dataloader(dataset_name, batch_size=32, num_workers=0, transform_data=None, only_test=False):
    if only_test:
        test_dataset = get_traditional_dataset(dataset_name, batch_size, num_workers, transform_data, only_test=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return None, None, test_loader, None 

    train_dataset, val_dataset, test_dataset = get_traditional_dataset(dataset_name, batch_size, num_workers, transform_data)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    classes = set([c for _, c in train_dataset])
    n_classes = len(classes)

    return train_loader, val_loader, test_loader, n_classes

def get_custom_dataloader(custom_dataset_class, data_path, batch_size=32, shuffle=True, num_workers=2, transforms=None):
    if transforms is None:
        transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
    
    dataset = custom_dataset_class(data_path, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_number_of_classes(dataset):
    pass 