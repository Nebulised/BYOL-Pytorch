import torch
import torchvision
import torchvision.datasets

DATASET_CHOICES = ["custom",
                   "emnist_by-class",
                   "emnist_by-merge",
                   "emnist_balanced",
                   "emnist_letters",
                   "emnist_digits",
                   "emnist_mnist",
                   "cifar10"]


def get_dataset(type, path,percent_data_to_use = 1.0, percent_train_to_use_as_val = 0.0,  train_transform = None, test_transform = None, **kwargs):
    val_dataset = None

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
                                                   transform=test_transform)
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

    if percent_train_to_use_as_val > 0.0:
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [1-percent_train_to_use_as_val, percent_train_to_use_as_val])


    if percent_data_to_use < 1.0:
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [percent_data_to_use, 1-percent_data_to_use])
        if val_dataset is not None:
            val_dataset, _ = torch.utils.data.random_split(val_dataset, [percent_data_to_use, 1-percent_data_to_use])
            val_dataset.transform = test_transform


    print(f"Total number of training samples : {len(train_dataset)} | Total number of validation samples : {0 if val_dataset is None else len(val_dataset)} | Total number of test samples : {len(test_dataset)}")
    return train_dataset, val_dataset, test_dataset