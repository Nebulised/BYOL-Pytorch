# Boostrap Your Own Latent (BYOL)
## Implemented in Pytorch

---
This is an unofficial pytorch implementation of the ["Bootstrap Your Own Latent
A New Approach to Self-Supervised Learning"](https://arxiv.org/pdf/2006.07733.pdf#section.4) paper.

The official repo by the authors can be found [here](https://github.com/deepmind/deepmind-research/tree/master/byol)

This repo is designed to be as close to the original tensorflow implementation as possible. 

---
### Implemented Features
* All augmentation types
  * Randomised colour jitter as per the paper
  * Proper colour dropping through PIL image grayscale conversion
* Exponential Moving Average(EMA) target network
  * Base EMA Tau -> 1.0 over course of pre-training
* Linear evaluation training
* Fine-tuning
  * Batch norm momentum setup
* Supports Emnist and CIFAR-10 datasets currently 
* Optional mlflow integration
* Lars optimiser 
  * Does not apply weight decay to batch norm and bias parameters
 ---
### Known Missing Features

---
### Validation Results
| Pre-Train Optimiser | Pre-Train LR |  Pre-Train Num Epochs  | Model Backbone | Linear Eval Optimiser | Linear Eval LR | Linear Eval Num Epochs | Test Accuracy |       Path to full params       |
|:-------------------:|:------------:|:----------------------:|:--------------:|:---------------------:|:--------------:|:----------------------:|:-------------:|:-------------------------------:|
|        Adam         |    0.0003    |          1000          |    Resnet18    |          SGD          |      0.1       |           25           |     0.91      | validation_experiments/cifar_10 |
---
### Self-Supervised Training
Example command to  train on the CIFAR-10 dataset on GPU 0 from scratch with 8 workers per dataloader
```commandline
python main.py --dataset-type "cifar10" --dataset-path "/path/to/dataset/" --model-output-folder-path "/path/to/save/folder/to/" --run-type "train" --gpu 0 --num-workers 8
```
---
### Fine-Tune
Example command to fine-tune on the CIFAR-10 dataset on GPU 0 from scratch with 8 workers per dataloader
```commandline
python main.py --dataset-type "cifar10" --dataset-path "/path/to/dataset/" --model-output-folder-path "/path/to/save/folder/to/" --run-type "fine_tune" --model-path "/path/to/pre-trained/model" --gpu 0 --num-workers 8
```

---
### Argument Info 

 * --run-type
   * Expected type : string
   * Optional? : NO
   * Choices
     * train
     * fine-tune
     * eval
   * ****
 * --model-output-folder-path
   * Expected type : string
   * Optional? : YES
   * If this argument is specified the model checkpoint files will be saved within this folder. Models will not be saved if this is not specified. 
 * --dataset-path 
   * Expected type : string
   * Optional? NO
   * Path to the dataset src folder (not the individual train/test/val folder)
 * --dataset-type
   * Expected type : string
   * Optional? NO
   * Choices
     * custom
     * emnist_by-class
     * emnist_by-merge
     * emnist_balanced
     * emnist_letters
     * emnist_digits
     * emnist_mnist
     * cifar10
   * Dataset type. If training on your own dataset use the "custom" option.
     * If using a custom dataset. Path specified must be a src folder containing "train" and "test" sub-folders.
     * Within these sub-folders are the classes. See torchvision image folder documentation for more info 
 * --model-path
   * Expected type : string
   * Optional? NO if run-type=="train" else YES
   * Path to the pre-trained model
 * --num-workers
   * Expected type : int 
   * Optional? YES
     * Defaults to 0
   * Nun workers for each dataloader (1 dataloader when self-supervised training, 2 dataloaders when fine-tuning )
   * When num_workers > 0 multiprocessing is used for the num workers specified
 * --gpu
   * Expected type : int
   * Optional? YES
   * Which GPU to run training/validation on
 * --mflow-tracking-uri
   * Expected type : string
   * Optional? YES
   * If specified mlflow logging/use is enabled
   * Sets mlflow tracking-uri to value passed for this arg
 * --mlflow-experiment-name
   * Expected type : string
   * Optional? YES
     * Defaults to "byol_experiment"
   * Sets mlflow experiment name to value passed
   * If mlflow tracking-uri is not specified this arg does nothing
 * --mlflow-run-id
   * Expected type : string
   * Optional? YES
   * Sets mlflow run id
   * Only to be used when fine-tuning or evaluating.
     * Automatically nests runs in mlflow if run id specified
 * --resume-training
   * Expected type : None (is called without an arg)
   * Optional? YES
   * Used to load model from a checkpoint and to continue training from that checkpointe epoch and optimiser state
* --model-param-file-path
  * * Expected type : string
  * Optional? YES
    * Defaults to parameters/model_params.yaml
  * Path to model params yaml file 
* --run-param-file-path
  * Optional? NO
  * Path to run param file path 
---

### Mlflow Integration

This repo is designed to log parameters, metrics and the models as artifacts.
This is only done if the set-tracking-uri argument is specified.
Otherwise, this repo can be used fine without mlflow installed.

 
---

### TODO
 - [X] Implement LARS Optimiser
 - [X] Validate results on CIFAR-10 dataset
 - [X] Implement Linear evaluation fine-tuning
 - [X] Implement logistic regression fine-tuning
 - [X] Implement resumable training
 - [X] Mlflow integration
 - [X] Docstring
 - [ ] Multi-GPU support
 - [X] Allow use with custom dataset types 

---

### Changelog
* V1.0
  * Implemented BYOL in pytorch validated against  Cifar-10 dataset
  * Has support for emnist and cifar-10 dataset