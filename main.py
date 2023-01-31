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


def get_params(path):
    with open(path,
              "r") as yaml_file:
        try:
            return yaml.safe_load(yaml_file)
        except yaml.YAMLError as yaml_error:
            print(yaml_error)




def main():
    args = get_args()


    run_type = args.run_type
    model_params = get_params("parameters/model_params.yaml")
    params = get_params(f"parameters/{run_type}_params.yaml")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    num_workers = args.num_workers
    print(f"Running on device : {device}")
    model = BYOL(max_num_steps = None,
                 **model_params).to(device)

    byol_augmenter = BYOLAugmenter(model_input_height= model.input_height,
                                   model_input_width=model.input_width)

    if run_type == "train":
        byol_augmenter.setup_multi_view(view_1_params=params["augmentation"]["view_1"],
                                        view_2_params=params["augmentation"]["view_2"])
        dataset, _, _ = get_dataset(type = args.dataset_type,
                                    train_transform = byol_augmenter.self_supervised_pre_train_transform,
                                    **params)
        data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                                  batch_size = args.batch_size,
                                                  shuffle = True,
                                                  num_workers = args.num_workers)
        train_model(device = device,
                    checkpoint_output_path=args.model_output_folder_path,
                    data_loader= data_loader,
                    **params)

    elif args.run_type == "fine-tune":
        test_params = get_params("parameters/test_params.yaml")
        train_dataset, val_dataset, _ = get_dataset(args.dataset_type,
                                                    train_transform = byol_augmenter.get_fine_tune_augmentations(**params["augmentation"]),
                                                    test_transform = byol_augmenter.get_test_augmentations(**test_params["augmentation"]),
                                                    **params)
        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=args.batch_size,
                                                        shuffle=True,
                                                        num_workers=args.num_workers)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                      batch_size=args.batch_size,
                                                      shuffle=True,
                                                      num_workers=args.num_workers)

        fine_tune(model = model,
                  device = device,
                  train_data_loader = train_data_loader,
                  val_data_loader = val_data_loader,
                  model_output_folder_path = args.model_output_folder_path,
                  **params)


    elif args.run_type == "eval":
        raise NotImplementedError("Eval mode not implemented yet")


def fine_tune(model,
              device,
              train_data_loader,
              val_data_loader,
              freeze_encoder,
              num_classes,
              learning_rate,
              momentum,
              num_epochs,
              print_every,
              checkpoint_every,
              model_output_folder_path,
              validate_every):
    freeze_encoder = freeze_encoder
    model.name = model.name + "_fine_tuned_"
    loss_function = torch.nn.CrossEntropyLoss()
    num_classes = num_classes
    encoder_model = model.online_encoder.to(device)
    if freeze_encoder:
        for param in encoder_model.parameters():
            param.requires_grad = False
        encoder_model.eval()
    model.create_fc(num_classes = num_classes)
    model.fc.to(device)

    optimiser = torch.optim.SGD(model.fc.parameters(),
                                lr = learning_rate,
                                momentum = momentum)
    lowest_val_loss = None
    for epoch_index in range(num_epochs):
        for minibatch_index, (images, labels) in enumerate(train_data_loader):
            images, labels = images.to(device), labels.to(device)
            model_output = model(images)
            loss = loss_function(model_output,
                                 labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if (minibatch_index + 1) % print_every == 0:
                print(f"Epoch {epoch_index} | Minibatch {minibatch_index} / {len(train_data_loader)} | Loss : {loss}")
        if (epoch_index + 1) % checkpoint_every == 0:
            model.save(model_output_folder_path,
                       optimiser = optimiser,
                       epoch = epoch_index)
        if (epoch_index + 1) % validate_every == 0:
            validation_loss = test(model = model,
                                   test_data_loader = val_data_loader,
                                   device = device)
            if lowest_val_loss is None or validation_loss < lowest_val_loss:
                model.save(folder_path = model,
                           epoch = epoch_index,
                           optimiser = optimiser,
                           model_save_name = "byol_model_fine_tuned_lowest_val.pt")
                lowest_val_loss = validation_loss



def train_model(model,   learning_rate, num_epochs, device, print_every,checpoint_every, checkpoint_output_path, data_loader):
    optimiser = torch.optim.Adam(model.get_all_online_params(),
                                 lr = learning_rate)


    model.set_max_num_steps(len(data_loader) * num_epochs)


    for epoch_index in range(num_epochs):
        epoch_start_time = time.time()
        for minibatch_index, ((view_1, view_2), _) in enumerate(data_loader):
            optimiser.zero_grad()
            loss = model(view_1.to(device),
                         view_2.to(device))
            loss.backward()
            optimiser.step()
            model.update_target_network()
            if (minibatch_index + 1) % print_every == 0: print(f"Epoch {epoch_index} | Minibatch {minibatch_index} / {len(train_data_loader)} | Loss : {loss} | Current tau : {model.current_tau}")
        print(f"Time taken for epoch : {time.time() - epoch_start_time}")
        if (epoch_index + 1) % checpoint_every == 0:
            model.save(folder_path = checkpoint_output_path,
                       epoch = epoch_index,
                       optimiser = optimiser)


def test(model,
         test_data_loader,
         device):
    model.eval()
    correct = 0
    total = 0
    loss_function = torch.nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        for images, labels in test_data_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            losses.append(loss_function(output, labels).item())
            _, predicted = torch.max(output.data,
                                     1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the {total} test images: {100 * correct // total} %')
    model.train()
    return sum(losses)/len(losses)


if __name__ == '__main__':
    main()
