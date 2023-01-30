import argparse
import copy
import time

import torch
import torchvision
import yaml

from augmentations import BYOLAugmenter
from networks import BYOL



DATASET_CHOICES = ["custom",
                   "emnist_by-class",
                   "emnist_by-merge",
                   "emnist_balanced",
                   "emnist_letters",
                   "emnist_digits",
                   "emnist_mnist",
                   "cifar10"]

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-output-folder-path",
                        type = str,
                        help = "Path to folder to save checkpoints to")
    parser.add_argument("--dataset-type",
                        type = str,
                        help = "Which dataset to train from. Can be a custom or emnist",
                        choices=DATASET_CHOICES,
                        required=True)
    parser.add_argument("--dataset-path",
                        type = str,
                        help="Path to location of dataset",
                        required=True)
    parser.add_argument("--run-type",
                        type = str,
                        choices = ["train","fine-tune","eval"],
                        required = True,
                        help = "Whether to train, fine-tune or eval")

    parser.add_argument("--model-path",
                        type = str)



    return parser.parse_args()

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
    training_params = params["train"]
    fine_tune_params = params["fine_tune"]
    test_params = params["test"]

    model = BYOL(max_num_steps = None,
                 **model_params).to(device)

    augmenter = BYOLAugmenter(view_1_params = training_params["augmentation"]["view_1"],
                              view_2_params = training_params["augmentation"]["view_2"],
                              fine_tune_params = fine_tune_params["augmentation"],
                              test_params = test_params["augmentation"],
                              model_input_width = model.input_width,
                              model_input_height = model.input_height)



    run_type = args.run_type

    if run_type == "train":
        train_model(device = device, args = args, training_params = training_params, model = model, augmenter=augmenter)

    elif args.run_type == "fine-tune":
        fine_tune(model = model, args = args, fine_tune_params = fine_tune_params, model_params = model_params, device = device, augmenter=augmenter)


    elif args.run_type == "eval":
        raise NotImplementedError("Eval mode not implemented yet")


# def get_fine_tune_transforms(model_input_height, model_input_width):
#     return torchvision.transforms.Compose([torchvision.transforms.RandomResizedCrop(size = (model_input_height, model_input_width)),
#                                                                                           torchvision.transforms.RandomHorizontalFlip(),
#                                                                                           torchvision.transforms.ToTensor(),
#                                                                                           torchvision.transforms.Normalize(0.5,
#                                                                                                                            0.25)])

def fine_tune(model, args, fine_tune_params, model_params, device, augmenter):
    model.load(args.model_path)
    loss_function = torch.nn.CrossEntropyLoss()
    train_dataset, test_dataset = get_dataset(type = args.dataset_type,
                                              path = args.dataset_path,
                                              train_transform = augmenter.test_augs,
                                              test_transform = augmenter.test_augs)

    dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size = fine_tune_params["batch_size"],
                                             shuffle = False,
                                             num_workers = fine_tune_params["num_workers"])

    encoder_model = copy.deepcopy(model.online_encoder).to(device)
    for param in encoder_model.parameters():
        param.requires_grad = False
    encoder_model.eval()
    fc = torch.nn.Linear(in_features = model_params["embedding_size"],
                         out_features = 10)



    full_model = torch.nn.Sequential(encoder_model, fc).to(device)
    optimiser = torch.optim.SGD(fc.parameters(),
                                lr = fine_tune_params["learning_rate"],
                                momentum = 0.9)


    for epoch_index in range(fine_tune_params["num_epochs"]):
        for minibatch_index, (images, labels) in enumerate(dataloader):
            images, labels = images.to(device), labels.to(device)
            model_output = full_model(images)
            loss = loss_function(model_output,
                                 labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if (minibatch_index + 1) % fine_tune_params["print_every"] == 0: print(f"Epoch {epoch_index} | Minibatch {minibatch_index} / {len(dataloader)} | Loss : {loss}")
        if (epoch_index + 1) % 15 == 0: test(model = full_model,
                                             test_dataset = test_dataset,
                                             device = device)







def train_model(model, device, args, training_params, augmenter):
    ### Model setup
    ############ Dataset setup ###########
    train_dataset, _ = get_dataset(type = args.dataset_type,
                                   path = args.dataset_path,
                                   train_transform = augmenter.self_supervised_pre_train_transform)
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size = training_params["batch_size"],
                                             shuffle = False,
                                             num_workers = training_params["num_workers"])
    optimiser = torch.optim.Adam(model.get_all_online_params(),
                                 lr = training_params["learning_rate"])

    num_epochs = training_params["num_epochs"]
    model.set_max_num_steps(len(dataloader) * num_epochs)

    for epoch_index in range(num_epochs):
        epoch_start_time = time.time()
        for minibatch_index, ((view_1, view_2), _) in enumerate(dataloader):
            optimiser.zero_grad()
            loss = model.forward(view_1.to(device),
                                 view_2.to(device))
            loss.backward()
            optimiser.step()
            model.update_target_network()
            if (minibatch_index + 1) % training_params["print_every"] == 0: print(f"Epoch {epoch_index} | Minibatch {minibatch_index} / {len(dataloader)} | Loss : {loss} | Current tau : {model.current_tau}")
        print(f"Time taken for epoch : {time.time() -epoch_start_time}")
        if (epoch_index + 1) % training_params["checkpoint_every"] == 0:
            model.save(folder_path = args.model_output_folder_path,
                       epoch = epoch_index,
                       optimiser = optimiser)



def test(model, test_dataset, device):
    model.eval()
    correct = 0
    total = 0

    test_data_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size = 1,
                                                   shuffle = False)
    with torch.no_grad():
        for images, labels in test_data_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data,1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {len(test_data_loader)} test images: {100 * correct // total} %')
    model.train()

if __name__ == '__main__':
    main()