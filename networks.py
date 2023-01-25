import copy

import torch.nn
import torchvision.transforms
from torchvision.models import get_model
from augmentations import *

ACTIVATION_OPTIONS = {
    "relu":torch.nn.ReLU,
    "leaky_relu":torch.nn.LeakyReLU,
    "none":torch.nn.Identity
    }



class BYOL(torch.nn.Module):

    def __init__(self,
                 encoder_model: str = "resnet50",
                 embedding_size: int = 128,
                 num_projection_layers: int = 1,
                 projection_size: int = 128,
                 num_predictor_layers: int = 1):
        super().__init__()
        self.embedding_size = embedding_size
        self.projection_size = projection_size

        self.online_encoder = get_model(name = encoder_model,
                                        weights = None,
                                        num_classes = self.embedding_size)

        self.online_projection_head = MLP(num_layers = num_projection_layers,
                                          size_in = self.embedding_size,
                                          size_out = self.projection_size,
                                          batch_norm_enabled = True)

        self.online_predictor = MLP(num_layers = num_predictor_layers,
                                    size_in = self.projection_size,
                                    size_out = self.projection_size,
                                    batch_norm_enabled = True)

        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_projection_head = copy.deepcopy(self.online_projection_head)

        self.online_network = torch.nn.Sequential(self.online_encoder,
                                                   self.online_projection_head,
                                                   self.online_predictor)
        self.target_network = torch.nn.Sequential(self.target_encoder,
                                                   self.target_projection_head)

        self.view_1_augmentations = torchvision.transforms.Compose([BYOLResizedCrop(image_aug_prob = 0.8),
                                                                    BYOLHorizontalFlip(image_aug_prob = 0.5),
                                                                    BYOLColorJitter(image_aug_prob = 0.8),
                                                                    BYOLGaussianBlur(image_aug_prob = 1.0),
                                                                    BYOLSolarisation(image_aug_prob = 0.0)])

        self.view_2_augmentations = torchvision.transforms.Compose([BYOLResizedCrop(image_aug_prob = 0.8),
                                                                    BYOLHorizontalFlip(image_aug_prob = 0.5),
                                                                    BYOLColorJitter(image_aug_prob = 0.8),
                                                                    BYOLGaussianBlur(image_aug_prob = 0.1),
                                                                    BYOLSolarisation(image_aug_prob = 0.2)])




    @staticmethod
    def regression_loss(predicted: torch.Tensor,
                        expected: torch.Tensor):
        return 2 - (2 * torch.nn.functional.cosine_similarity(x1 = predicted,
                                                              x2 = expected))

    def get_image_views(self, image):
        image_view_1 = self.view_1_augmentations(image.clone())
        image_view_2 = self.view_2_augmentations(image.clone())
        return image_view_1, image_view_2



    def forward(self, image_1, image_2, inference  = False):
        online_output_1 = self.online_network(image_1)
        online_output_2 = self.online_network(image_2)
        with torch.no_grad():
            target_output_1 = self.target_network(image_1).detach()
            target_output_2 = self.target_network(image_2).detach()

        total_loss = self.regression_loss(online_output_1, target_output_1) + self.regression_loss(online_output_2,
                                                                                                   target_output_2)

        return total_loss




class LinearLayer(torch.nn.Module):

    def __init__(self,
                 size_in: int,
                 size_out: int,
                 bias: bool = True,
                 batch_norm_enabled: bool = True,
                 activation="relu"):
        super().__init__()

        if activation not in ACTIVATION_OPTIONS:
            raise ValueError(f"Expected activation function to be one of {list(ACTIVATION_OPTIONS.keys())}. Received"
                             f" {activation}")
        self.activation = ACTIVATION_OPTIONS[activation]()
        self.batch_norm = torch.nn.BatchNorm2d(num_features = size_out) if batch_norm_enabled else torch.nn.Identity()
        self.linear = torch.nn.Linear(in_features = size_in,
                                      out_features = size_out,
                                      bias = bias)

    def forward(self,
                input_matrix):
        output = self.linear(input_matrix)
        output = self.batch_norm(output)
        output = self.activation(output)
        return output


class MLP(torch.nn.Module):

    def __init__(self,
                 num_layers: int,
                 size_in: int,
                 size_out: int,
                 batch_norm_enabled: bool):
        super(MLP,
              self).__init__()
        layers = [LinearLayer(size_in = size_in if layer_index == 0 else size_out,
                              size_out = size_out,
                              activation = "relu",
                              batch_norm_enabled = batch_norm_enabled) for layer_index in range(num_layers - 1)]
        layers.append(LinearLayer(size_in = size_out,
                                  size_out = size_out,
                                  activation = "none",
                                  batch_norm_enabled = False))

        self.all_layers = torch.nn.Sequential(*layers)

    def forward(self,
                input_vector):
        return self.all_layers(input_vector)
