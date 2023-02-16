import argparse
import os
import time

import mlflow
import torch

from augmentations import BYOLAugmenter
from dataset import get_dataset, DATASET_CHOICES
from networks import BYOL
from utils import get_params, log_param_dicts, TrainingTracker


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
                        choices=["train", "fine-tune", "eval"],
                        required=True,
                        help="Whether to train, fine-tune or eval")
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
    parser.add_argument("--model-param-file-path",
                        type = str,
                        help = "Path to model params yaml file",
                        default = "parameters/model_params.yaml")
    parser.add_argument("--run-param-file-path",
                        type = str,
                        help = "Path to train/fine-tune/inference params yaml file")
    parser.add_argument("--mlflow-run-id",
                        type = str,
                        help="Mlflow run id to either resume training or to nest run under")

    parsed_args = parser.parse_args()
    # Model path must be specified when fine tuning or testing
    if parsed_args.run_type != "train":
        assert parsed_args.model_path is not None, "The '--model-path' argument must be specified when fine-tuning, performing linear evaluation or inference"

    return parsed_args


def main():
    args = get_args()
    run_type = args.run_type
    model_params = get_params(args.model_param_file_path)
    params = get_params(args.run_param_file_path)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")


    ### Setting up mlflow if required
    mlflow_tracking_uri = args.mlflow_tracking_uri
    if mlflow_tracking_uri is not None:
        print(f"Mlflow enabled. Tracking URI : {mlflow_tracking_uri}")
        import mlflow
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment_name)
        if run_type == "train":
            nested = False
            run_name = "Self-Supervised-Pre-Training"
        elif run_type == "fine-tune":
            nested = True
            run_name = "Fine-Tuning"
        else:
            nested = True
            run_name = "Evaluation"
        if nested:
            mlflow.start_run(run_id=args.mlflow_run_id)
        mlflow.start_run(nested = nested,
                         run_name = run_name)
        mlflow_enabled = True
        log_param_dicts(param_dict=params)
        log_param_dicts(param_dict=model_params,
                        existing_key="model")
        # Vars converts namespace object to dict
        log_param_dicts(param_dict=vars(args))
        for path_to_param_file in (args.model_param_file_path, args.run_param_file_path):
            mlflow.log_artifact(local_path = path_to_param_file,
                                artifact_path = "parameters")
    else:
        mlflow_enabled = False

    print(f"Running on device : {device}")
    model = BYOL(max_num_steps=None,
                 **model_params).to(device)
    if args.model_path is not None : model.load(args.model_path)

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
    elif args.run_type == "fine-tune":
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

    if mlflow_enabled:
        mlflow.end_run()



def fine_tune(model: BYOL,
              device: torch.device,
              train_data_loader: torch.utils.data.DataLoader,
              val_data_loader: torch.utils.data.DataLoader,
              freeze_encoder: bool,
              num_classes: int,
              optimiser_params : dict,
              num_epochs: int,
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
    metric_tracker = TrainingTracker(mlflow_enabled=mlflow_enabled)

    ### Setting up model
    encoder_model = model.online_encoder.to(device)
    if freeze_encoder:
        for param in encoder_model.parameters():
            param.requires_grad = False
        encoder_model.eval()

    # Setting up model classification output layer
    model.create_fc(num_classes=num_classes)
    model.fc.to(device)

    optimiser = torch.optim.Adam(model.fc.parameters() if freeze_encoder else torch.nn.Sequential(encoder_model, model.fc).parameters(),
                                **optimiser_params)
    lowest_val_loss = None
    training_start_time = time.time()
    # Fine tuning
    for epoch_index in range(num_epochs):
        epoch_start_time = time.time()
        for minibatch_index, (images, labels) in enumerate(train_data_loader):
            images, labels = images.to(device), labels.to(device)
            model_output = model(images)
            loss = loss_function(model_output,
                                 labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            metric_tracker.log_metric("Train Loss", loss.item())

        if (epoch_index + 1) % checkpoint_every == 0:
            saved_model_path = model.save(checkpoint_output_folder_path,
                                          optimiser = optimiser,
                                          epoch = epoch_index)
            if mlflow_enabled : mlflow.log_artifact(saved_model_path,"checkpoints")

        ### Validate trained model
        if (epoch_index + 1) % validate_every == 0:
            validation_loss, val_acc = test(model = model,
                                            test_data_loader = val_data_loader,
                                            device = device)
            metric_tracker.log_metric("Validation Loss", validation_loss)
            metric_tracker.log_metric("Validation Accuracy", val_acc)

            ### Save lowest validation loss model
            if lowest_val_loss is None or validation_loss < lowest_val_loss:
                model_save_path = model.save(folder_path = checkpoint_output_folder_path,
                                             epoch = epoch_index,
                                             optimiser = optimiser,
                                             model_save_name = "byol_model_fine_tuned_lowest_val.pt")
                lowest_val_loss = validation_loss
                if mlflow_enabled : mlflow.log_artifact(model_save_path, "checkpoints")
        metric_tracker.increment_epoch()
        epoch_elapsed_time = time.time() - epoch_start_time
        training_elapsed_time = time.time() - training_start_time
        expected_seconds_till_completion = (training_elapsed_time / (epoch_index + 1)) * (num_epochs - epoch_index - 1)
        print(f"Time taken for epoch : {elapsed_to_hms(epoch_elapsed_time)} |  Estimated time till completion : {elapsed_to_hms(expected_seconds_till_completion)}")

def train_model(model,
                optimiser_params,
                num_epochs,
                device,
                checkpoint_every,
                checkpoint_output_folder_path,
                data_loader,
                mlflow_enabled=False,
                cosine_annealing=False,
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

    optimiser = torch.optim.Adam(torch.nn.Sequential(model.online_encoder, model.online_projection_head, model.online_predictor).parameters(),
                                **optimiser_params)

    if cosine_annealing : scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimiser,
                                                                                 T_max = num_epochs,
                                                                                 eta_min = 0.)
    training_start_time = time.time()
    model.set_max_num_steps(len(data_loader) * num_epochs)
    metric_tracker = TrainingTracker(mlflow_enabled = mlflow_enabled)
    for epoch_index in range(num_epochs):
        epoch_start_time = time.time()
        for minibatch_index, ((view_1, view_2), _) in enumerate(data_loader):
            loss = model(view_1.to(device),
                         view_2.to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            model.update_target_network()
            metric_tracker.log_metric("Train Loss", loss.item())
            metric_tracker.log_metric("Ema Tau", model.current_tau)

        if cosine_annealing:
            current_lr = scheduler.get_last_lr()[0]
            metric_tracker.log_metric("Scheduler LR", current_lr)
            scheduler.step()


        if (epoch_index + 1) % checkpoint_every == 0:
            model_save_path = model.save(folder_path = checkpoint_output_folder_path,
                                         epoch = epoch_index,
                                         optimiser = optimiser)

            if mlflow_enabled : mlflow.log_artifact(model_save_path, "checkpoints")
        metric_tracker.increment_epoch()
        epoch_elapsed_time = time.time() - epoch_start_time
        training_elapsed_time = time.time() - training_start_time
        expected_seconds_till_completion = (training_elapsed_time / (epoch_index + 1)) * (num_epochs - (epoch_index + 1))
        print(f"Time taken for epoch : {elapsed_to_hms(epoch_elapsed_time)} |  Estimated time till completion : {elapsed_to_hms(expected_seconds_till_completion)}")

def elapsed_to_hms(elapsed_time):
    return time.strftime('%H:%M:%S:', time.gmtime(elapsed_time))

def test(model : BYOL,
         test_data_loader : torch.utils.data.DataLoader,
         device : torch.device):
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
            losses.append(loss_function(output, labels).detach().item())
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
