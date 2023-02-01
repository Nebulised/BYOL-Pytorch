import argparse
import time

import mlflow
import torch

from augmentations import BYOLAugmenter
from dataset import get_dataset, DATASET_CHOICES
from networks import BYOL
from utils import get_params, log_param_dicts


def get_args():
    """
    Argparser method


    Returns:
        namespace of arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-output-folder-path",
                        type=str,
                        help="Path to folder to save checkpoints to")
    parser.add_argument("--dataset-type",
                        type=str,
                        help="Which dataset type to use",
                        choices=DATASET_CHOICES,
                        required=True)
    parser.add_argument("--dataset-path",
                        type=str,
                        help="Path to src folder of dataset",
                        required=True)
    parser.add_argument("--run-type",
                        type=str,
                        choices=["train", "fine_tune", "eval"],
                        required=True,
                        help="Whether to train, fine_tune or eval")
    parser.add_argument("--gpu",
                        type=int,
                        help="Which GPU to run on")
    parser.add_argument("--model-path",
                        type=str,
                        help="path to model to load in to fine tune, test or resume training on ")
    parser.add_argument("--num-workers",
                        type=int,
                        default=0,
                        help="Num workers for dataloaders")
    parser.add_argument("--mlflow-tracking-uri",
                        type=str,
                        help = "Tracking URI of mlflow. If specified Mlflow is enabled")
    parser.add_argument("--mlflow-experiment-name",
                        type=str,
                        default="byol_experiment",
                        help = "Name of experiemnt to save mlflow run under")

    parsed_args = parser.parse_args()
    # Model path must be specifide when fine tuning or testing
    if parsed_args.run_type != "train":
        assert parsed_args.model_path is not None, "The '--model-path' argument must be specified when fine-tuning, performing linear evaluation or inference"

    return parsed_args


def main():
    args = get_args()
    run_type = args.run_type
    # Loads in param files
    model_params = get_params("parameters/model_params.yaml")
    params = get_params(f"parameters/{run_type}_params.yaml")
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


    ### Setting up mlflow if required
    mlflow_tracking_uri = args.mlflow_tracking_uri
    if mlflow_tracking_uri is not None:
        print(f"Mlflow enabled. Tracking URI : {mlflow_tracking_uri}")
        import mlflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
        mlflow.start_run()
        mlflow_enabled = True
        log_param_dicts(param_dict=params)
        log_param_dicts(param_dict=model_params,
                        existing_key="model")
    else:
        mlflow_enabled = False

    print(f"Running on device : {device}")
    model = BYOL(max_num_steps=None,
                 **model_params).to(device)

    byol_augmenter = BYOLAugmenter(resize_output_height=model.input_height,
                                   resize_output_width=model.input_width)


    ### Self-Supervised Training
    if run_type == "train":
        byol_augmenter.setup_multi_view(view_1_params=params["augmentation"]["view_1"],
                                        view_2_params=params["augmentation"]["view_2"])
        dataset, _, _ = get_dataset(type=args.dataset_type,
                                    path=args.dataset_path,
                                    train_transform=byol_augmenter.self_supervised_pre_train_transform,
                                    **params)
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=params["batch_size"],
                                                  shuffle=True,
                                                  num_workers=args.num_workers)
        train_model(model=model,
                    mlflow_enabled=mlflow_enabled,
                    device=device,
                    checkpoint_output_folder_path=args.model_output_folder_path,
                    data_loader=data_loader,
                    **params)



    # Supervised fine tuning
    elif args.run_type == "fine_tune":
        test_params = get_params("parameters/test_params.yaml")
        train_dataset, val_dataset, _ = get_dataset(type=args.dataset_type,
                                                    path=args.dataset_path,
                                                    train_transform=byol_augmenter.get_fine_tune_augmentations(
                                                        **params["augmentation"]),
                                                    test_transform=byol_augmenter.get_test_augmentations(
                                                        **test_params["augmentation"]),
                                                    **params)

        train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                        batch_size=params["batch_size"],
                                                        shuffle=True,
                                                        num_workers=args.num_workers)
        val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                      batch_size=test_params["batch_size"],
                                                      shuffle=True,
                                                      num_workers=args.num_workers)

        fine_tune(model=model,
                  device=device,
                  train_data_loader=train_data_loader,
                  val_data_loader=val_data_loader,
                  mlflow_enabled=mlflow_enabled,
                  checkpoint_output_folder_path=args.model_output_folder_path,
                  **params)


    elif args.run_type == "eval":
        raise NotImplementedError("Eval mode not implemented yet")

    if mlflow_enabled: mlflow.end_run()


def fine_tune(model: BYOL,
              device: torch.device,
              train_data_loader: torch.utils.data.DataLoader,
              val_data_loader: torch.utils.data.DataLoader,
              freeze_encoder: bool,
              num_classes: int,
              learning_rate: float,
              momentum: float,
              num_epochs: int,
              print_every: int,
              checkpoint_every: int,
              checkpoint_output_folder_path: str,
              validate_every: int,
              mlflow_enabled: bool = False,
              **kwargs):
    """ Fine-tuning method

    Args:
        model:
            Model to train, model must already be loaded
        device:
            Device to put model/data on to
        train_data_loader:
            Data loader for training dataset
            Must return a batch of images
        val_data_loader:
            Data loader for validation dataset
            Must return a batch of images
        freeze_encoder:
            Whether to freeze the encoder section
            If True : Equivalent to linear evaluatin
            If False : Equivalent to logistic regression
        num_classes:
            Num different classes of labels
            Corresponds to output size of inference model
        learning_rate:
            Learning rate for optimiser
        momentum:
            Momentum for sgd optimiser
        num_epochs:
            Num epochs to fine tune for
        print_every:
            How often to print training info
        checkpoint_every:
            How often to save checkpoint models
        checkpoint_output_folder_path:
            Where to output checkpoint models to
        validate_every:
            How often to perform validation
        mlflow_enabled:
            Whether to use mlflow integration
        **kwargs:
            Redundant args

    Returns:
        None
    """
    freeze_encoder = freeze_encoder
    model.name = model.name + "_fine_tuned_"
    loss_function = torch.nn.CrossEntropyLoss()
    num_classes = num_classes

    ### Setting up model
    encoder_model = model.online_encoder.to(device)
    if freeze_encoder:
        for param in encoder_model.parameters():
            param.requires_grad = False
        encoder_model.eval()

    # Setting up model classification output layer
    model.create_fc(num_classes=num_classes)
    model.fc.to(device)

    optimiser = torch.optim.SGD(model.fc.parameters(),
                                lr=learning_rate,
                                momentum=momentum)
    lowest_val_loss = None

    # Fine tuning
    for epoch_index in range(num_epochs):
        losses = []
        for minibatch_index, (images, labels) in enumerate(train_data_loader):
            images, labels = images.to(device), labels.to(device)
            model_output = model(images)
            loss = loss_function(model_output,
                                 labels)
            losses.append(loss.item())
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            if (minibatch_index + 1) % print_every == 0:
                print(f"Epoch {epoch_index} | Minibatch {minibatch_index} / {len(train_data_loader)} | Loss : {loss}")
        if mlflow_enabled: mlflow.log_metric("Fine-Tune Loss",
                                             sum(losses) / len(losses),
                                             step=epoch_index)
        if (epoch_index + 1) % checkpoint_every == 0:
            model.save(checkpoint_output_folder_path,
                       optimiser=optimiser,
                       epoch=epoch_index)

        # Perform validation model
        if (epoch_index + 1) % validate_every == 0:
            validation_loss, acc = test(model=model,
                                        test_data_loader=val_data_loader,
                                        device=device)
            if mlflow_enabled:
                mlflow.log_metric("Validation loss",
                                  validation_loss.item(),
                                  step=epoch_index)
                mlflow.log_metric("Validation acc",
                                  acc,
                                  step=epoch_index)
            if lowest_val_loss is None or validation_loss < lowest_val_loss:
                model.save(folder_path=checkpoint_output_folder_path,
                           epoch=epoch_index,
                           optimiser=optimiser,
                           model_save_name="byol_model_fine_tuned_lowest_val.pt")
                lowest_val_loss = validation_loss


def train_model(model,
                learning_rate,
                num_epochs,
                device,
                print_every,
                checkpoint_every,
                checkpoint_output_folder_path,
                data_loader,
                mlflow_enabled=False,
                **kwargs):
    """Self supervised training method

    Args:
        model:
            Model to train, if resumin training model must already be loaded
        learning_rate:
            Learning rate for optimiser
        num_epochs:
            Num epochs to train for
        device:
            Device to put model/data on to
        print_every:
            How often to print training info
        checkpoint_every:
            How often to save checkpoint models
        checkpoint_output_folder_path:
            Where to output checkpoint models to
        data_loader:
            Data loader for training data
            Must return two differing augmented views of same original image
       mlflow_enabled:
            Whether to use mlflow integration

        **kwargs:

    Returns:
        None
    """
    optimiser = torch.optim.Adam(model.get_all_online_params(),
                                 lr=learning_rate)

    model.set_max_num_steps(len(data_loader) * num_epochs)

    for epoch_index in range(num_epochs):
        epoch_start_time = time.time()
        losses = []
        for minibatch_index, ((view_1, view_2), _) in enumerate(data_loader):
            optimiser.zero_grad()
            loss = model(view_1.to(device),
                         view_2.to(device))
            losses.append(loss.item())
            loss.backward()
            optimiser.step()
            model.update_target_network()
            if (minibatch_index + 1) % print_every == 0: print(
                f"Epoch {epoch_index} | Minibatch {minibatch_index} / {len(data_loader)} | Loss : {loss} | Current tau : {model.current_tau}")
        print(f"Time taken for epoch : {time.time() - epoch_start_time}")

        #Log mlflow appropriate data
        if mlflow_enabled:
            mlflow.log_metric("Train Loss",
                              sum(losses) / len(losses),
                              step=epoch_index)
            mlflow.log_metric("Ema Tau",
                              model.current_tau)
        if (epoch_index + 1) % checkpoint_every == 0:
            # Save model checkpoints
            model.save(folder_path=checkpoint_output_folder_path,
                       epoch=epoch_index,
                       optimiser=optimiser)


def test(model,
         test_data_loader,
         device):
    """Method to test/perform inference using model

    Args:
        model:
            Model to perform validation on, must already be loaded
        test_data_loader:
            Data loader for test data
            Must return a batch of single images
        device:
            Device to put model/data on to

    Returns:
        float : average loss
        float : model accuracy
    """
    # Convert model to eval mode
    model.eval()
    correct = 0
    total = 0
    loss_function = torch.nn.CrossEntropyLoss()
    losses = []
    with torch.no_grad():
        for images, labels in test_data_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            losses.append(loss_function(output,
                                        labels).item())
            _, predicted = torch.max(output.data,
                                     1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = correct / total
    print(f'Accuracy of the network on the {total} test images: {100 * acc} %')

    ### Reset model to train
    model.train()
    return sum(losses) / len(losses), acc


if __name__ == '__main__':
    main()
