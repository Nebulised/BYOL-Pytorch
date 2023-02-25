import argparse
import os
import time

import mlflow
import torch

from augmentations import BYOLAugmenter
from dataset import get_dataset, DATASET_CHOICES
from networks import BYOL
from utils import get_params, TrainingTracker, CosineAnnealingLRWithWarmup, setup_mlflow, create_optimiser


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
    parser.add_argument("--resume-training",
                        action="store_true")
    parser.add_argument("--num-workers",
                        type=int,
                        default=0,
                        help="Num workers for dataloaders")
    parser.add_argument("--model-param-file-path",
                        type=str,
                        help="Path to model params yaml file",
                        default="parameters/model_params.yaml")
    parser.add_argument("--run-param-file-path",
                        type=str,
                        required=True,
                        help="Path to train/fine-tune/inference params yaml file")
    parser.add_argument("--mlflow-tracking-uri",
                        type=str,
                        help="Tracking URI of mlflow. If specified Mlflow is enabled")
    parser.add_argument("--mlflow-experiment-name",
                        type=str,
                        default="byol_experiment",
                        help="Name of experiemnt to save mlflow run under")
    parser.add_argument("--mlflow-run-id",
                        type=str,
                        help="Mlflow run id to either resume training or to nest run under")

    parsed_args = parser.parse_args()
    # Model path must be specified when fine-tuning or testing
    if parsed_args.run_type != "train":
        assert parsed_args.model_path is not None, "The '--model-path' argument must be specified when fine-tuning, performing linear evaluation or inference"

    return parsed_args


def main():
    args = get_args()
    run_type = args.run_type
    model_params = get_params(args.model_param_file_path)
    run_params = get_params(args.run_param_file_path)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    optimiser_state_dict, start_epoch = None, 0

    ### Setting up mlflow if required
    if args.mlflow_tracking_uri is not None:
        mlflow_enabled = True
        setup_mlflow(run_params=run_params,
                     model_params=model_params,
                     args=args,
                     **vars(args))
    else:
        mlflow_enabled = False
    optimiser_params = run_params["optimiser_params"]

    print(f"Running on device: {device}")
    model = BYOL(**model_params)

    model.to(device)
    if run_type in ("train", "fine-tune"):
        if args.resume_training:
            optimiser_state_dict, start_epoch = model.load(args.model_path)
            print(f"Resuming training. Existing optimiser state dict will be used.  Starting training from epoch {start_epoch}")
        freeze_encoder = None
        if run_type == "fine-tune":
            #  If not resuming training instantiate a linear output layer
            freeze_encoder = run_params["freeze_encoder"]
            if not args.resume_training:
                model.create_fc(run_params["num_classes"])
                print("Not resuming training. Output linear layer created")
            # If fine-tuning rather than linear eval divide the weight decay by learning rate as per the paper
            if not run_params["freeze_encoder"]:
                optimiser_params["weight_decay"] /= optimiser_params["lr"]
                print(f"Dividing weight decay by their learning rate. New weight decay : {optimiser_params['weight_decay']}. Check the BYOL paper for why")
        # All models have a metric tracker and optimiser
        metric_tracker = TrainingTracker(mlflow_enabled=mlflow_enabled)
        optimiser = create_optimiser(model=model,
                                     optimiser_params=optimiser_params,
                                     optimiser_state_dict=optimiser_state_dict,
                                     run_type=run_type,
                                     freeze_encoder=freeze_encoder)


    if run_type == "train":
        scheduler = CosineAnnealingLRWithWarmup(optimiser=optimiser,
                                                warmup_epochs=run_params["warmup_epochs"],
                                                num_epochs_total=run_params["num_epochs"],
                                                last_epoch=-1 if start_epoch == 0 else start_epoch,
                                                verbose=False,
                                                cosine_eta_min=0.0) if run_params["cosine_annealing"] else None

        pre_train(model=model,
                  optimiser=optimiser,
                  metric_tracker=metric_tracker,
                  start_epoch=start_epoch,
                  scheduler=scheduler,
                  device=device,
                  **vars(args),
                  **run_params)



    # Supervised fine tuning
    elif args.run_type == "fine-tune":

        fine_tune(model=model,
                  device=device,
                  metric_tracker=metric_tracker,
                  start_epoch=start_epoch,
                  optimiser=optimiser,
                  **vars(args),
                  **run_params)


    elif args.run_type == "eval":
        eval(model=model,
             device=device,
             **vars(args),
             **run_params)

    if mlflow_enabled:
        mlflow.end_run()


def eval(model: BYOL,
         dataset_type: str,
         dataset_path: str,
         augmentation_params: dict,
         num_workers: int,
         device: torch.device,
         mlflow_enabled: bool = False,
         **kwargs):
    """
    Args:
        device:
            Device to put data on to
        augmentation_params:
            Augmentation parameters. A dict containing the "test" key
        num_workers:
            Num workers for data loader
        model:
            model. Should be pre-loaded
        dataset_type:
            Dataset type to perform evaluatin on
        dataset_path:
            Path to src folder on dataset
        mlflow_enabled:
            Whether to enable mlflow integration

    Returns:
        None
    """
    byol_augmenter = BYOLAugmenter(resize_output_height=model.input_height,
                                   resize_output_width=model.input_width)

    _, _, test_dataset = get_dataset(type=dataset_type,
                                     path=dataset_path,
                                     test_transform=byol_augmenter.get_test_augmentations(**augmentation_params["test"]))
    test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                   batch_size=1,
                                                   shuffle=False,
                                                   num_workers=num_workers)
    average_loss, accuracy = test(model=model,
                                  test_data_loader=test_data_loader,
                                  device=device)
    print(f"Average loss : {average_loss} | Test accuracy : {accuracy}")
    if mlflow_enabled:
        mlflow.log_metric("Average loss",
                          value=average_loss)
        mlflow.log_metric("Accuracy",
                          value=accuracy)


def fine_tune(model: BYOL,
              optimiser: torch.optim.Optimizer,
              num_epochs: int,
              device: torch.device,
              checkpoint_every: int,
              model_output_folder_path: str,
              dataset_type: str,
              dataset_path: str,
              freeze_encoder: bool,
              augmentation_params: dict,
              num_workers: int,
              metric_tracker: TrainingTracker,
              batch_size: int,
              percent_train_to_use_as_val: float,
              percent_data_to_use: float,
              validate_every: int,
              start_epoch: int = 0,
              mlflow_enabled: bool = False,
              **kwargs):
    """ Fine-tuning method

    Args:
        start_epoch:
            Epoch to start training from. If resuming training this shoul be non zero
        percent_data_to_use:
            Amount of total data to use for training/validation as a float(0.0 - 1.0) representing a percentage
        percent_train_to_use_as_val:
            Amount of training data to use as validation data as a float(0.0 - 1.0) representing a percentage if no validation data available
        batch_size:
            Batch size for training
        metric_tracker:
            TrainingTracker object to record and display current training info
        num_workers:
            Num workers per dataloader
        augmentation_params:
            Parameters for augmentation.
            Should contain a "train"  and "test" key corresponding to the appropriate augmentation details
        dataset_path:
            Path to src folder of dataset
        dataset_type:
            Type of dataset
        optimiser:
            Pytorch optimiser
        model:
            Model to train, model must already be loaded
        device:
            Device to put model/data on to
        freeze_encoder:
            Whether to freeze the encoder section
            If True : Equivalent to linear evaluation
            If False : Equivalent to logistic regression
        num_epochs:
            Num epochs to fine tune for
        checkpoint_every:
            How often to save checkpoint models
        model_output_folder_path:
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

    byol_augmenter = BYOLAugmenter(resize_output_height=model.input_height,
                                   resize_output_width=model.input_width)

    train_dataset, val_dataset, _ = get_dataset(type=dataset_type,
                                                path=dataset_path,
                                                train_transform=byol_augmenter.get_fine_tune_augmentations(**augmentation_params["train"]),
                                                test_transform=byol_augmenter.get_test_augmentations(**augmentation_params["test"]),
                                                percent_data_to_use=percent_data_to_use,
                                                percent_train_to_use_as_val=percent_train_to_use_as_val)

    train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                    batch_size=batch_size,
                                                    shuffle=True,
                                                    num_workers=num_workers)
    val_data_loader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=True,
                                                  num_workers=num_workers)

    model.name = model.name + "_fine_tuned_"
    loss_function = torch.nn.CrossEntropyLoss()
    if freeze_encoder:
        model.online_encoder.eval()
        for param in model.online_encoder.parameters():
            param.requires_grad = False
    if not freeze_encoder:
        for layer in model.encoder_model.modules():
            if isinstance(layer,
                          torch.nn.BatchNorm2d):
                # As per the paper
                layer.momentum = max(1 - 10 / (len(train_data_loader)),
                                     0.9)

    lowest_val_loss = None
    training_start_time = time.time()
    # Fine tuning
    for epoch_index in range(start_epoch,
                             num_epochs):
        epoch_start_time = time.time()
        for minibatch_index, (images, labels) in enumerate(train_data_loader):
            images, labels = images.to(device), labels.to(device)
            model_output = model(images)
            loss = loss_function(model_output,
                                 labels)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            metric_tracker.log_metric("Train Loss",
                                      loss.item())

        if (epoch_index + 1) % checkpoint_every == 0:
            saved_model_path = model.save(model_output_folder_path,
                                          optimiser=optimiser,
                                          epoch=epoch_index)
            if mlflow_enabled: mlflow.log_artifact(saved_model_path,
                                                   "checkpoints")

        ### Validate trained model
        if (epoch_index + 1) % validate_every == 0:
            validation_loss, val_acc = test(model=model,
                                            test_data_loader=val_data_loader,
                                            device=device)
            metric_tracker.log_metric("Validation Loss",
                                      validation_loss)
            metric_tracker.log_metric("Validation Accuracy",
                                      val_acc)

            ### Save lowest validation loss model
            if lowest_val_loss is None or validation_loss < lowest_val_loss:
                model_save_path = model.save(folder_path=model_output_folder_path,
                                             epoch=epoch_index,
                                             optimiser=optimiser,
                                             model_save_name="byol_model_fine_tuned_lowest_val.pt")
                lowest_val_loss = validation_loss
                if mlflow_enabled: mlflow.log_artifact(model_save_path,
                                                       "checkpoints")
        metric_tracker.increment_epoch()
        epoch_elapsed_time = time.time() - epoch_start_time
        training_elapsed_time = time.time() - training_start_time
        expected_seconds_till_completion = (training_elapsed_time / (epoch_index + 1)) * (num_epochs - (epoch_index + 1))
        print(f"Time taken for epoch : {elapsed_to_hms(epoch_elapsed_time)} |  Estimated time till completion : {elapsed_to_hms(expected_seconds_till_completion)}")


def pre_train(model: BYOL,
              optimiser: torch.optim.Optimizer,
              num_epochs: int,
              device: torch.device,
              checkpoint_every: int,
              model_output_folder_path: int,
              dataset_type: str,
              dataset_path: str,
              scheduler,
              num_workers: int,
              metric_tracker: TrainingTracker,
              batch_size: int,
              augmentation_params: dict,
              percent_data_to_use: float,
              start_epoch=0,
              mlflow_enabled=False,
              **kwargs):
    """Self supervised training method

    Args:
        scheduler:
            Learning rate scheduler. if none scheduler will not be used
        percent_data_to_use:
            Percentage of total training data to use
        augmentation_params:
            Parameters for augmentation.
            Should contain a "view_1"  and "view_2" key corresponding to the two views appropriate augmentation details
        batch_size:
            Batch size for training
        metric_tracker:
            TrainingTracker object to record and display current training info
        num_workers:
            Num workers per data loader
        dataset_path:
            Path to src folder of dataset
        dataset_type:
            Type of dataset
        optimiser:
            Pytorch optimiser
        start_epoch:
            Epoch to resume training from. If resuming should be non-zero
        model:
            Model to train, if resuming training model must already be loaded
        num_epochs:
            Num epochs to train for
        device:
            Device to put model/data on to
        checkpoint_every:
            How often to save checkpoint models
        model_output_folder_path:
            Where to output checkpoint models to
       mlflow_enabled:
            Whether to use mlflow integration

        **kwargs:

    Returns:
        None
    """
    byol_augmenter = BYOLAugmenter(resize_output_height=model.input_height,
                                   resize_output_width=model.input_width)
    byol_augmenter.setup_multi_view(view_1_params=augmentation_params["view_1"],
                                    view_2_params=augmentation_params["view_2"])
    dataset, _, _ = get_dataset(type=dataset_type,
                                path=dataset_path,
                                train_transform=byol_augmenter.self_supervised_pre_train_transform,
                                percent_train_to_use_as_val=0.0,
                                percent_data_to_use=percent_data_to_use)
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers)

    training_start_time = time.time()
    model.set_max_num_steps(len(data_loader) * num_epochs)
    for epoch_index in range(start_epoch,
                             num_epochs):
        epoch_start_time = time.time()
        for minibatch_index, ((view_1, view_2), _) in enumerate(data_loader):
            loss = model(view_1.to(device),
                         view_2.to(device))
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            model.update_target_network()
            metric_tracker.log_metric("Train Loss",
                                      loss.item())
            metric_tracker.log_metric("Ema Tau",
                                      model.current_tau)

        if scheduler is not None:
            current_lr = scheduler.get_last_lr()[0]
            metric_tracker.log_metric("Scheduler LR",
                                      current_lr)
            scheduler.step()

        if (epoch_index + 1) % checkpoint_every == 0:
            model_save_path = model.save(folder_path=model_output_folder_path,
                                         epoch=epoch_index,
                                         optimiser=optimiser)

            if mlflow_enabled: mlflow.log_artifact(model_save_path,
                                                   "checkpoints")
        metric_tracker.increment_epoch()
        epoch_elapsed_time = time.time() - epoch_start_time
        training_elapsed_time = time.time() - training_start_time
        expected_seconds_till_completion = (training_elapsed_time / (epoch_index + 1)) * (num_epochs - (epoch_index + 1))
        print(f"Time taken for epoch : {elapsed_to_hms(epoch_elapsed_time)} |  Estimated time till completion : {elapsed_to_hms(expected_seconds_till_completion)}")


def elapsed_to_hms(elapsed_time):
    return time.strftime('%H:%M:%S:',
                         time.gmtime(elapsed_time))


def test(model: BYOL,
         test_data_loader: torch.utils.data.DataLoader,
         device: torch.device):
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
                                        labels).detach().item())
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
