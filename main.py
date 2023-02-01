import argparse
import copy
import time

import mlflow
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
                        choices = ["train", "fine_tune", "eval"],
                        required = True,
                        help = "Whether to train, fine-tune or eval")
    parser.add_argument("--gpu",
                        type = int)
    parser.add_argument("--model-path",
                        type = str)
    parser.add_argument("--num-workers",
                        type = int,
                        default = 0)
    parser.add_argument("--mlflow-tracking-uri",
                        type = str)
    parser.add_argument("--mlflow-experiment-name",
                        type = str,
                        default = "byol_experiment")


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


def log_param_dicts(param_dict, existing_key = None):

    for key, val in param_dict.items():
        current_concat_key = f"{existing_key}_{key}" if existing_key is not None else key
        if type(val) is dict:
            log_param_dicts(val, existing_key = current_concat_key)
        else:
            if "every" not in current_concat_key:
                mlflow.log_param(current_concat_key, val)


def main():
    args = get_args()


    run_type = args.run_type
    model_params = get_params("parameters/model_params.yaml")
    params = get_params(f"parameters/{run_type}_params.yaml")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    mlflow_tracking_uri = args.mlflow_tracking_uri
    if mlflow_tracking_uri is not None:
        print(f"Mlflow enabled. Tracking URI : {mlflow_tracking_uri}")
        import mlflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
        mlflow.start_run()
        mlflow_enabled = True
        log_param_dicts(param_dict = params)
        log_param_dicts(param_dict = model_params, existing_key = "model")
    else:
        mlflow_enabled = False

    print(f"Running on device : {device}")
    model = BYOL(max_num_steps = None,
                 **model_params).to(device)
    if args.model_path is not None : model.load(args.model_path)

    byol_augmenter = BYOLAugmenter(model_input_height= model.input_height,
                                   model_input_width=model.input_width)

    if run_type == "train":
        byol_augmenter.setup_multi_view(view_1_params=params["augmentation"]["view_1"],
                                        view_2_params=params["augmentation"]["view_2"])
        dataset, _, _ = get_dataset(type = args.dataset_type,
                                    path = args.dataset_path,
                                    train_transform = byol_augmenter.self_supervised_pre_train_transform,
                                    **params)
        data_loader = torch.utils.data.DataLoader(dataset = dataset,
                                                  batch_size = params["batch_size"],
                                                  shuffle = True,
                                                  num_workers = args.num_workers)
        train_model(model = model,
                    mlflow_enabled = mlflow_enabled,
                    device = device,
                    checkpoint_output_path=args.model_output_folder_path,
                    data_loader= data_loader,
                    **params)




    elif args.run_type == "fine_tune":
        test_params = get_params("parameters/test_params.yaml")
        train_dataset, val_dataset, _ = get_dataset(type = args.dataset_type,
                                                    path = args.dataset_path,
                                                    train_transform = byol_augmenter.get_fine_tune_augmentations(**params["augmentation"]),
                                                    test_transform = byol_augmenter.get_test_augmentations(**test_params["augmentation"]),
                                                    **params)

        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=params["batch_size"],
                                                        shuffle=True,
                                                        num_workers=args.num_workers)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                      batch_size=test_params["batch_size"],
                                                      shuffle=True,
                                                      num_workers=args.num_workers)

        fine_tune(model = model,
                  device = device,
                  train_data_loader = train_data_loader,
                  val_data_loader = val_data_loader,
                  mlflow_enabled = mlflow_enabled,
                  model_output_folder_path = args.model_output_folder_path,
                  **params)


    elif args.run_type == "eval":
        raise NotImplementedError("Eval mode not implemented yet")

    if mlflow_enabled : mlflow.end_run()


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
              validate_every,
              mlflow_enabled = False,
              **kwargs):
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
        losses = []
        for minibatch_index, (images, labels) in enumerate(train_data_loader):
            images, labels = images.to(device), labels.to(device)
            model_output = model(images)
            loss = loss_function(model_output,
                                 labels)
            losses.append(loss.detach().item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if (minibatch_index + 1) % print_every == 0:
                print(f"Epoch {epoch_index} | Minibatch {minibatch_index} / {len(train_data_loader)} | Loss : {loss}")
        if mlflow_enabled: mlflow.log_metric("Fine-Tune Loss",
                                             sum(losses)/len(losses),
                                             step = epoch_index)
        if (epoch_index + 1) % checkpoint_every == 0:
            saved_model_path = model.save(model_output_folder_path,
                                          optimiser = optimiser,
                                          epoch = epoch_index)
            if mlflow_enabled : mlflow.log_artifact(saved_model_path,"checkpoints")
        if (epoch_index + 1) % validate_every == 0:
            validation_loss, acc = test(model = model,
                                   test_data_loader = val_data_loader,
                                   device = device)
            if mlflow_enabled :
                mlflow.log_metric("Validation loss", validation_loss, step = epoch_index)
                mlflow.log_metric("Validation acc", acc, step = epoch_index)
            if lowest_val_loss is None or validation_loss < lowest_val_loss:
                model_save_path = model.save(folder_path = model_output_folder_path,
                                             epoch = epoch_index,
                                             optimiser = optimiser,
                                             model_save_name = "byol_model_fine_tuned_lowest_val.pt")
                lowest_val_loss = validation_loss
                if mlflow_enabled : mlflow.log_artifact(model_save_path, "checkpoints")



def train_model(model, learning_rate, num_epochs, device, print_every,checkpoint_every, checkpoint_output_path, data_loader,mlflow_enabled = False, **kwargs):
    optimiser = torch.optim.Adam(model.get_all_online_params(),
                                 lr = learning_rate)


    model.set_max_num_steps(len(data_loader) * num_epochs)


    for epoch_index in range(num_epochs):
        epoch_start_time = time.time()
        losses = []
        for minibatch_index, ((view_1, view_2), _) in enumerate(data_loader):
            optimiser.zero_grad()
            loss = model(view_1.to(device),
                         view_2.to(device))
            losses.append(loss.detach().item())
            loss.backward()
            optimiser.step()
            model.update_target_network()
            if (minibatch_index + 1) % print_every == 0: print(f"Epoch {epoch_index} | Minibatch {minibatch_index} / {len(data_loader)} | Loss : {loss} | Current tau : {model.current_tau}")
        print(f"Time taken for epoch : {time.time() - epoch_start_time}")
        if mlflow_enabled:
            mlflow.log_metric("Train Loss",
                              sum(losses)/len(losses),
                              step = epoch_index)
            mlflow.log_metric("Ema Tau", model.current_tau)
        if (epoch_index + 1) % checkpoint_every == 0:
            model_save_path = model.save(folder_path = checkpoint_output_path,
                                         epoch = epoch_index,
                                         optimiser = optimiser)

            if mlflow_enabled : mlflow.log_artifact(model_save_path, "checkpoints")



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
            losses.append(loss_function(output, labels).detach().item())
            _, predicted = torch.max(output.data,
                                     1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(f'Accuracy of the network on the {total} test images: {100 * acc} %')
    model.train()
    return sum(losses)/len(losses), acc


if __name__ == '__main__':
    main()
