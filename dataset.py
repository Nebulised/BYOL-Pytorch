import os

import sklearn.model_selection
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


def get_dataset(type : str,
                path : str,
                percent_data_to_use : float=1.0,
                percent_train_to_use_as_val : float=0.0,
                train_transform=None,
                test_transform=None,
                **kwargs):
    """Method for getting train/val/test datasets

    Args:
        type:
            Type of dataset
            One of those in DATASET_CHOICES constant
        path:
            Path to src folder of dataset
        percent_data_to_use:
            Percent of data to use for training/validation set
        percent_train_to_use_as_val:
            Perent of training data to use as validation data
        train_transform:
            Transforms to apply to train data
        test_transform:
            Transforms to apply to val/test data
        **kwargs:

    Returns:
        train dataset, val dataset, test dataset
    """
    val_dataset = None

    if "emnist" in type:
        _, split = type.split("_")
        train_dataset = torchvision.datasets.EMNIST(root=path,
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
        train_dataset = torchvision.datasets.CIFAR10(root=path,
                                                     train=True,
                                                     download=False,
                                                     transform=train_transform)
        test_dataset = torchvision.datasets.CIFAR10(root=path,
                                                    train=False,
                                                    download=False,
                                                    transform=test_transform)



    elif type == "custom":
        train_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path, "train"),
                                                         transform=train_transform)
        test_dataset = torchvision.datasets.ImageFolder(root=os.path.join(path, "test"),
                                                        transform=test_transform)
    else:
        raise ValueError(f"Invalid dataset type. Expected one of : {DATASET_CHOICES}")

    if percent_data_to_use < 1.0:
        train_dataset = torch.utils.data.Subset(train_dataset,
                                                sklearn.model_selection.train_test_split(torch.arange(len(train_dataset)),
                                                                                         train_size=percent_data_to_use,
                                                                                         stratify=train_dataset.targets,
                                                                                         random_state=42)[0])

    if percent_train_to_use_as_val > 0.0:
        train_split_indexes, val_split_indexes = sklearn.model_selection.train_test_split(torch.arange(len(train_dataset)),
                                                                                          train_size=percent_train_to_use_as_val,
                                                                                          test_size=1-percent_train_to_use_as_val,
                                                                                          stratify=[train_dataset.dataset.targets[index] for index in train_dataset.indices],
                                                                                          random_state=42)

        new_train_dataset = torch.utils.data.Subset(train_dataset,
                                                    train_split_indexes)
        val_dataset = torch.utils.data.Subset(train_dataset,
                                              val_split_indexes)
        train_dataset = new_train_dataset


    # if percent_data_to_use < 1.0 and val_dataset is not None:
    #     val_dataset = torch.utils.data.Subset(val_dataset,
    #                                           sklearn.model_selection.train_test_split(torch.arange(len(val_dataset)),
    #                                                                                    train_size=percent_data_to_use,
    #                                                                                    stratify=[val_dataset.dataset.dataset.targets[index] for index in val_dataset.dataset.indices],
    #                                                                                    random_state=42)[0])
    if val_dataset is not None : val_dataset.transform = test_transform

    for each_dataset, dataset_name in [(train_dataset, "Training Data"), (val_dataset, "Val Data"), (test_dataset, "Test Data")]:
        if each_dataset is None:
            continue
        if isinstance(each_dataset, torch.utils.data.Subset):
            targets = [each_dataset.dataset.dataset.targets[index] for index in each_dataset.dataset.indices] if isinstance(each_dataset.dataset, torch.utils.data.Subset) else [each_dataset.dataset.targets[index] for index in each_dataset.indices]
        else:
            targets = each_dataset.targets
        classes, number_each_class = torch.unique(torch.LongTensor(targets), return_counts=True)
        count_dict = {k : v for k,v in zip(classes.tolist(), number_each_class.tolist())}
        print(f"Total number of Samples for {dataset_name} : {len(each_dataset)} | Number of samples per class : {count_dict}")

    return train_dataset, val_dataset, test_dataset
