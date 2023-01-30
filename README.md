# Boostrap Your Own Latent (BYOL)
## Implemented in Pytorch

---
This is a pytorch implementation of the ["Bootstrap Your Own Latent
A New Approach to Self-Supervised Learning"](https://arxiv.org/pdf/2006.07733.pdf#section.4) paper.

The official repo by the authors can be found [here](https://github.com/deepmind/deepmind-research/tree/master/byol)

This repo is designed to be as close to the original implementation as possible. 

---
### Implemented Features
* All augmentation types
  * Randomised colour jitter as per the paper
* Exponential Moving Average(EMA) target network
  * Base EMA Tau -> 1.0 over course of pre training
* Linear evaluation training 
* Supports Emnist and CIFAR-10 datasets currently 
 ---
### Known Missing Features
* Lars optimiser
* Full fine-tuning

---
### Self-Supervised Training
Example command to  train on the CIFAR-10 dataset 
```commandline
python main.py --dataset-type "cifar10" --dataset-path "/path/to/dataset/" --model-output-folder-path "/path/to/save/folder/to/" --run-type "train"
```
####
Arguments
 * --run-type
   * Expected type : string
   * Optional? : NO
   * Choices
     * train
     * fine-tune
     * eval
   * **This must be set to "train" for self-supervised training**
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

---
### Linear Evaluation 
```commandline
python main.py --dataset-type "cifar10" --dataset-path "/path/to/dataset/" --model-output-folder-path "/path/to/save/folder/to/" --run-type "train" --model-path "/path/to/pre-trained/model"
```
 * --run-type
   * Expected type : string
   * Optional? : NO
   * Choices
     * train
     * fine-tune
     * eval
   * **This must be set to "fine-tune" for linear evaluation training**
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
 * --model-path
   * Expected type : string
   * Optional? NO 
   * Path to the pre-trained model

### TODO
* Lars Optimiser
* Validate results on CIFAR-10 dataset
* Implement full fine-tuning
* Implement resumable training
* Log/Graph results
* Docstring 
* Multi-GPU support 