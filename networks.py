import copy
import os

import torch.nn
import torchvision.transforms
from torchvision.models import get_model
from augmentations import *
from torchvision.transforms import RandomApply, RandomResizedCrop, RandomHorizontalFlip, GaussianBlur, \
    RandomSolarize, ToTensor

ACTIVATION_OPTIONS = {
    "relu":torch.nn.ReLU,
    "leaky_relu":torch.nn.LeakyReLU,
    "none":torch.nn.Identity
    }


class BYOL(torch.nn.Module):

    def __init__(self,
                 augmentation_params: dict,
                 encoder_model: str = "resnet50",
                 embedding_size: int = 128,
                 num_projection_layers: int = 1,
                 projection_size: int = 128,
                 num_predictor_layers: int = 1,
                 input_height: int = 224,
                 input_width: int = 224,
                 projection_hidden_layer_size : int = 128,
                 ema_tau : int = 0.996,
                 name : str = "byol_model"):
        super().__init__()
        self.aug_params = augmentation_params
        self.name = name
        #TODO: Implement tau updating
        self.current_tau = ema_tau
        self.embedding_size = embedding_size
        self.projection_size = projection_size
        self.input_height = input_height
        self.input_width = input_width

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

        for param in self.target_network.parameters():
            param.requires_grad = False

        self.view_1_augs, self.view_2_augs = self.get_augmentations_compositions()

    @torch.no_grad()
    def update_target_network(self):
        online_network_state_dict = self.online_network.state_dict()
        target_network_state_dict = self.target_network.state_dict()
        for name, param in self.target_network.named_parameters():
            target_network_state_dict[name] += ((1-self.current_tau) * (online_network_state_dict[name] - param))


    def save(self, folder_path, epoch, optimiser):
        torch.save({'epoch':epoch,
                    'online_network_state_dict':self.online_network.state_dict(),
                    "target_network_state_dict": self.target_network.state_dict(),
                    'optimizer_state_dict':optimiser.state_dict(),
            }, os.path.join(folder_path, f"{self.name}_epoch={epoch}.pt"))

    def get_augmentations_compositions(self):
        augmentations = []
        for view in self.aug_params.values():
            c_j_params = view["colour_jitter"]
            gauss_blur_params = view["gaussian_blur"]
            solarization_params = view["solarization"]
            view_augmentations = torchvision.transforms.Compose([RandomApply([RandomResizedCrop(size = (self.input_height,self.input_width))],
                                                        p = view["random_crop"]),
                                            RandomHorizontalFlip(p = view["random_flip"]),
                                            RandomApply([BYOLColorJitter(max_brightness = c_j_params["brightness_adjustment"],
                                                                         max_contrast = c_j_params["contrast_adjustment"],
                                                                         max_saturation = c_j_params["saturation_adjustment"],
                                                                         max_hue = c_j_params["hue_adjustment"])],
                                                        p = c_j_params["probability"]),
                                            RandomApply([GaussianBlur(kernel_size = gauss_blur_params["kernel_size"],
                                                                      sigma = gauss_blur_params["std"])],
                                                                      p = gauss_blur_params["probability"]),
                                            RandomApply([RandomSolarize(threshold = solarization_params["threshold"],
                                                                        p = solarization_params["probability"])]),
                                                                 ToTensor()])
            augmentations.append(view_augmentations)
        return augmentations


    @ staticmethod
    def regression_loss(predicted: torch.Tensor,
                        expected: torch.Tensor):
        return 2 - (2 * torch.nn.functional.cosine_similarity(x1 = predicted,
                                                              x2 = expected))

    def get_image_views(self,
                        image):
        return self.view_1_augs(image), self.view_2_augs(image)

    def forward(self,
                image_1,
                image_2,
                inference=False):
        online_output_1 = self.online_network(image_1)
        online_output_2 = self.online_network(image_2)
        with torch.no_grad():
            target_output_1 = self.target_network(image_1).detach()
            target_output_2 = self.target_network(image_2).detach()

        total_loss = self.regression_loss(online_output_1,
                                          target_output_1) + self.regression_loss(online_output_2,
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
