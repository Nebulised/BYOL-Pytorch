import argparse
import time

import torch
import torchvision
import yaml
from networks import BYOL



DATASET_CHOICES = ["custom",
                   "emnist_by-class",
                   "emnist_by-merge",
                   "emnist_balanced",
                   "emnist_letters",
                   "emnist_digits",
                   "emnist_mnist"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-output-path",
                        type = str,
                        help = "Path to save model")
    parser.add_argument("--dataset-type",
                        type = str,
                        help = "Which dataset to train from. Can be a custom or emnist",
                        choices=DATASET_CHOICES,
                        required=True)
    parser.add_argument("--dataset-path",
                        type = str,
                        help="Path to location of dataset",
                        required=True)



    return parser.parse_args()

def get_dataset(type, path, transform = None):


    if "emnist" in type:
        _, split = type.split("_")
        train_dataset =  torchvision.datasets.EMNIST(root=path,
                                                     split=split,
                                                     train=True,
                                                     download=False,
                                                     transform=transform)
        test_dataset = torchvision.datasets.EMNIST(root=path,
                                                   split=split,
                                                   train=False,
                                                   download=False,
                                                   transform=transform), torchvision

    elif type == "custom":
        raise NotImplementedError("Custom datasets are not yet supported")
    else:
        raise ValueError(f"Invalid dataset type. Expected one of : {DATASET_CHOICES}")

    return train_dataset, test_dataset

def get_params():
    with open("params.yaml", "r") as yaml_file:
        try:
            return yaml.safe_load(yaml_file)
        except yaml.YAMLError as yaml_error:
            print(yaml_error)

def main():
    args = get_args()
    params = get_params()
    device = torch.device(f"cuda:{params['gpu']}" if torch.cuda.is_available() else "cpu")
    print(f"Running on device : {device}")

    model_params = params["model"]
    training_params = params["training"]
    aug_params = training_params["augmentation"]

    ### Model setup
    model = BYOL(augmentation_params = aug_params,
                 **model_params).to(device)

    ############ Dataset setup ###########
    train_dataset, test_dataset = get_dataset(type = args.dataset_type,
                                              path = args.dataset_path,
                                              transform = model.get_image_views)
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=training_params["batch_size"],
                                             shuffle=False,
                                             num_workers=training_params["num_workers"])
    optimiser = torch.optim.SGD(model.parameters(),lr = 0.001)
    for iteration_index in range(training_params["num_epochs"]):
        for i, ((view_1, view_2), _) in enumerate(dataloader):
            loss = model.forward(view_1.repeat(1,3,1,1).to(device), view_2.repeat(1,3,1,1).to(device),inference =
            False).mean()
            loss.backward()
            optimiser.step()
            print(f"Epoch {iteration_index} {i} / {len(dataloader)} | Loss : {loss}")






if __name__ == '__main__':
    main()