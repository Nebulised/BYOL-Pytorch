import argparse
import time

import torch
import torchvision

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
    parser.add_argument("--learning-rate",
                        type = float,
                        help = "Model learning rate",
                        required = True)
    parser.add_argument("--iterations",
                        type = int,
                        help = "Iteration",
                        required = True)
    parser.add_argument("--batch-size",
                        type = int,
                        help = "Batch size",
                        required = True)
    parser.add_argument("--model-output-path",
                        type = str,
                        help = "Path to save model")
    parser.add_argument("--dataset",
                        type = str,
                        help = "Which dataset to train from. Can be a custom or emnist",
                        choices=DATASET_CHOICES,
                        required=True)
    parser.add_argument("--dataset-path",
                        type = str,
                        help="Path to location of dataset",
                        required=True)



    return parser.parse_args()

def get_dataset(type, path, image_size=(224, 224)):
    transforms = [torchvision.transforms.Resize(size=image_size),
                  torchvision.transforms.ToTensor()]

    if "emnist" in type:
        _, split = type.split("_")
        train_dataset =  torchvision.datasets.EMNIST(root=path,
                                                     split=split,
                                                     train=True,
                                                     download=False,
                                                     transform=torchvision.transforms.Compose(transforms))
        test_dataset = torchvision.datasets.EMNIST(root=path,
                                                   split=split,
                                                   train=False,
                                                   download=False,
                                                   transform=torchvision.transforms.Compose(transforms)), torchvision

    elif type == "custom":
        raise NotImplementedError("Custom datasets are not yet supported")
    else:
        raise ValueError(f"Invalid dataset type. Expected one of : {DATASET_CHOICES}")

    return train_dataset, test_dataset



def main():
    args = get_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device : {device}")
    model = BYOL().to(device)

    ############ Dataset setup ###########
    train_dataset, test_dataset = get_dataset(type = args.dataset,
                                              path = args.dataset_path,
                                              image_size = (32, 32))
    dataloader = torch.utils.data.DataLoader(train_dataset,
                                             batch_size=args.batch_size,
                                             shuffle=True)
    optimiser = torch.optim.SGD(model.parameters(),lr = 0.001)
    for iteration_index in range(args.iterations):
        for i, (original_image, _) in enumerate(dataloader):
            start_time_to_image_to_device = time.time()
            original_image = original_image.repeat(1,3,1,1)
            print(f"Copy to device : {time.time() - start_time_to_image_to_device}")
            start_augment = time.time()
            image_view_1, image_view_2 = model.get_image_views(original_image)
            print(f"Augment to get views : {time.time() - start_augment}")
            start_loss = time.time()
            loss = model.forward(image_view_1.to(device), image_view_2.to(device),inference = False).mean()
            print(f"Time to calc loss {time.time() - start_loss}")

            start_backprop = time.time()
            loss.backward()
            optimiser.step()
            print(f" Back prop time {time.time() - start_backprop}")
            print(f"Epoch {iteration_index} {i} / {len(dataloader)} | Loss : {loss}")






if __name__ == '__main__':
    main()