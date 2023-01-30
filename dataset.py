import torchvision

DATASET_CHOICES = ["custom",
                   "emnist_by-class",
                   "emnist_by-merge",
                   "emnist_balanced",
                   "emnist_letters",
                   "emnist_digits",
                   "emnist_mnist",
                   "cifar10"]


def get_dataset(type, path, train_transform = None, test_transform = None):


    if "emnist" in type:
        _, split = type.split("_")
        train_dataset =  torchvision.datasets.EMNIST(root=path,
                                                     split=split,
                                                     train=True,
                                                     download=False,
                                                     transform=train_transform)
        test_dataset = torchvision.datasets.EMNIST(root=path,
                                                   split=split,
                                                   train=False,
                                                   download=False,
                                                   transform=torchvision.transforms.Compose([torchvision.transforms.Resize(64),
                                                                                             torchvision.transforms.ToTensor(),
                                                                                             torchvision.transforms.Normalize(0.5, 0.25)]))
    elif type == "cifar10":
        train_dataset = torchvision.datasets.CIFAR10(root = path,
                                                     train = True,
                                                     download = False,
                                                     transform = train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root = path,
                                                   train = False,
                                                   download = False,
                                                   transform = test_transform)



    elif type == "custom":
        raise NotImplementedError("Custom datasets are not yet supported")
    else:
        raise ValueError(f"Invalid dataset type. Expected one of : {DATASET_CHOICES}")

    return train_dataset, test_dataset