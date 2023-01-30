import argparse
import copy
import time

import torch
import torchvision
import yaml

from augmentations import BYOLAugmenter
from dataset import get_dataset, DATASET_CHOICES
from networks import BYOL


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-output-folder-path",
                        type = str,
                        help = "Path to folder to save checkpoints to")
    parser.add_argument("--dataset-type",
                        type = str,
                        help = "Which dataset to train from. Can be a custom or emnist",
                        choices = DATASET_CHOICES,
                        required = True)
    parser.add_argument("--dataset-path",
                        type = str,
                        help = "Path to location of dataset",
                        required = True)
    parser.add_argument("--run-type",
                        type = str,
                        choices = ["train", "fine-tune", "eval"],
                        required = True,
                        help = "Whether to train, fine-tune or eval")

    parser.add_argument("--model-path",
                        type = str)
    parser.add_argument("--num-workers",
                        type = int,
                        default = 0)


    parsed_args = parser.parse_args()
    if parsed_args.run_type != "train":
        assert parsed_args.model_path is not None, "The '--model-path' argument must be specified when fine-tuning, performing linear evaluation or inference"

    return parsed_args


def get_params():
    with open("params.yaml",
              "r") as yaml_file:
        try:
            return yaml.safe_load(yaml_file)
        except yaml.YAMLError as yaml_error:
            print(yaml_error)




def main():
    args = get_args()
    params = get_params()
    device = torch.device(f"cuda:{params['gpu']}" if torch.cuda.is_available() else "cpu")
    num_workers = args.num_workers
    print(f"Running on device : {device}")

    model_params = params["model"]
    training_params = params["train"]
    fine_tune_params = params["fine_tune"]
    test_params = params["test"]
    run_type = args.run_type
    if run_type == "train":
        pre_training = True
        current_params = training_params
    else:
        pre_training = False
        current_params = fine_tune_params




    model = BYOL(max_num_steps = None,
                 **model_params).to(device)



    batch_size = current_params["batch_size"]

    augmenter = BYOLAugmenter(view_1_params = training_params["augmentation"]["view_1"],
                              view_2_params = training_params["augmentation"]["view_2"],
                              fine_tune_params = fine_tune_params["augmentation"],
                              test_params = test_params["augmentation"],
                              model_input_width = model.input_width,
                              model_input_height = model.input_height)






    train_dataset, val_dataset, test_dataset = get_dataset(type = args.dataset_type,
                                                           path = args.dataset_path,
                                                           train_transform = augmenter.self_supervised_pre_train_transform if pre_training else augmenter.fine_tune_augs,
                                                           test_transform = augmenter.test_augs,
                                                           percent_data_to_use = current_params["use_percent_train_data"],
                                                           percent_train_to_use_as_val = 0 if pre_training else current_params["use_percent_train_data_as_val"])



    train_data_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True,
                                                    num_workers = num_workers)
    val_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                  batch_size = batch_size,
                                                  shuffle = False,
                                                  num_workers = num_workers) if val_dataset is not None else None
    test_data_loader = torch.utils.data.DataLoader(val_dataset,
                                                   batch_size = 1,
                                                   shuffle = False,
                                                   num_workers = num_workers) if test_dataset is not None else None

    print(f"Batch size : {batch_size}")


    if run_type == "train":
        train_model(device = device,
                    args = args,
                    training_params = training_params,
                    model = model,
                    train_data_loader = train_data_loader)

    elif args.run_type == "fine-tune":
        fine_tune(model = model,
                  args = args,
                  fine_tune_params = fine_tune_params,
                  model_params = model_params,
                  device = device,
                  train_data_loader = train_data_loader,
                  val_data_loader = val_data_loader)

    elif args.run_type == "eval":
        raise NotImplementedError("Eval mode not implemented yet")


def fine_tune(model,
              args,
              fine_tune_params,
              model_params,
              device,
              train_data_loader,
              val_data_loader):
    model.load(args.model_path)
    loss_function = torch.nn.CrossEntropyLoss()

    encoder_model = copy.deepcopy(model.online_encoder).to(device)
    for param in encoder_model.parameters():
        param.requires_grad = False
    encoder_model.eval()
    fc = torch.nn.Linear(in_features = model_params["embedding_size"],
                         out_features = fine_tune_params["num_classes"])

    full_model = torch.nn.Sequential(encoder_model,
                                     fc).to(device)
    optimiser = torch.optim.SGD(fc.parameters(),
                                lr = fine_tune_params["learning_rate"],
                                momentum = fine_tune_params["momentum"])

    for epoch_index in range(fine_tune_params["num_epochs"]):
        for minibatch_index, (images, labels) in enumerate(train_data_loader):
            images, labels = images.to(device), labels.to(device)
            model_output = full_model(images)
            loss = loss_function(model_output,
                                 labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if (minibatch_index + 1) % fine_tune_params["print_every"] == 0: print(f"Epoch {epoch_index} | Minibatch {minibatch_index} / {len(val_data_loader)} | Loss : {loss}")
        if (epoch_index + 1) % 15 == 0: test(model = full_model,
                                             test_data_loader = val_data_loader,
                                             device = device)


def train_model(model,
                device,
                args,
                training_params,
                train_data_loader):
    optimiser = torch.optim.Adam(model.get_all_online_params(),
                                 lr = training_params["learning_rate"])

    num_epochs = training_params["num_epochs"]
    model.set_max_num_steps(len(train_data_loader) * num_epochs)

    for epoch_index in range(num_epochs):
        epoch_start_time = time.time()
        for minibatch_index, ((view_1, view_2), _) in enumerate(train_data_loader):
            optimiser.zero_grad()
            loss = model.forward(view_1.to(device),
                                 view_2.to(device))
            loss.backward()
            optimiser.step()
            model.update_target_network()
            if (minibatch_index + 1) % training_params["print_every"] == 0: print(f"Epoch {epoch_index} | Minibatch {minibatch_index} / {len(train_data_loader)} | Loss : {loss} | Current tau : {model.current_tau}")
        print(f"Time taken for epoch : {time.time() - epoch_start_time}")
        if (epoch_index + 1) % training_params["checkpoint_every"] == 0:
            model.save(folder_path = args.model_output_folder_path,
                       epoch = epoch_index,
                       optimiser = optimiser)


def test(model,
         test_data_loader,
         device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_data_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            _, predicted = torch.max(output.data,
                                     1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
    model.train()


if __name__ == '__main__':
    main()
