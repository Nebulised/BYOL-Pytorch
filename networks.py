import copy
import math
import os

import torch.nn
import torchvision.transforms
from torchvision.models import get_model
from augmentations import *
from torchvision.transforms import RandomApply, RandomResizedCrop, RandomHorizontalFlip, GaussianBlur, \
    RandomSolarize, ToTensor, Normalize, InterpolationMode

ACTIVATION_OPTIONS = {
    "relu":torch.nn.ReLU,
    "leaky_relu":torch.nn.LeakyReLU,
    "none":torch.nn.Identity
    }


class BYOL(torch.nn.Module):

    def __init__(self,
                 max_num_steps,
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
        self.name = name
        self.fc = None
        self.base_tau = ema_tau
        self.current_tau = ema_tau
        self.current_step = 0
        self.max_num_steps = max_num_steps
        self.embedding_size = embedding_size
        self.projection_size = projection_size
        self.input_height = input_height
        self.input_width = input_width

        self.online_encoder = get_model(name = encoder_model,
                                        weights = None)
        self.online_encoder.fc = torch.nn.Identity()

        self.online_projection_head = MLP(num_hidden_layers = num_projection_layers,
                                          size_in = self.embedding_size,
                                          hidden_layer_size = projection_hidden_layer_size,
                                          size_out = self.projection_size,
                                          batch_norm_enabled = True)

        self.online_predictor = MLP(num_hidden_layers = num_projection_layers,
                                    size_in = self.projection_size,
                                    size_out = self.projection_size,
                                    hidden_layer_size = projection_hidden_layer_size,
                                    batch_norm_enabled = True)

        self.target_encoder, self.target_projection_head = self.create_target_network()

    def create_target_network(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        target_projection_head = copy.deepcopy(self.online_projection_head)

        for param in list(target_encoder.parameters()) + list(target_projection_head.parameters()):
            param.requires_grad = False
        return target_encoder, target_projection_head


    def update_tau(self):
        assert self.max_num_steps is not None, "For EMA the max number of training steps must be specified"
        self.current_tau = 1 - (1-self.base_tau) * (math.cos((math.pi * self.current_step)/self.max_num_steps) + 1)/2

    def set_max_num_steps(self, new_max_num_steps):
        self.max_num_steps = new_max_num_steps

    def update_target_network(self):
        self.current_step += 1
        self.update_tau()
        with torch.no_grad():
            for online_encoder_param, target_encoder_param in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
                target_encoder_param.data = target_encoder_param.data * self.current_tau + (1-self.current_tau) * online_encoder_param.data
            for online_projector_param, target_projector_param in zip(self.online_projection_head.parameters(), self.target_projection_head.parameters()):
                target_projector_param.data = target_projector_param.data * self.current_tau + (1-self.current_tau) * online_projector_param.data


    def save(self, folder_path, epoch, optimiser, model_save_name = None):
        if model_save_name is None:
            model_save_name = f"{self.name}_epoch={epoch}.pt"
        save_path = os.path.join(folder_path, model_save_name)
        print(f"Saving model to {save_path}")
        torch.save({"epoch":epoch,
                    "online_encoder_state_dict" : self.online_encoder.state_dict(),
                    "online_projection_head_state_dict" : self.online_projection_head.state_dict(),
                    "online_predictor_state_dict" : self.online_predictor.state_dict(),
                    "target_encoder_state_dict" : self.target_encoder.state_dict(),
                    "target_projection_head_state_dict" : self.target_projection_head.state_dict(),
                    "optimiser_state_dict" : optimiser.state_dict(),
                    "current_step" : self.current_step,
                    "base_tau" : self.base_tau,
                    "fc" : self.fc

            }, save_path)

    def load(self, model_path):
        print(f"Loading model from : {model_path}")
        checkpoint = torch.load(model_path)
        self.online_encoder.load_state_dict(checkpoint["online_encoder_state_dict"])
        self.online_projection_head.load_state_dict(checkpoint["online_projection_head_state_dict"])
        self.online_predictor.load_state_dict(checkpoint["online_predictor_state_dict"])
        self.target_encoder.load_state_dict(checkpoint["target_encoder_state_dict"])
        self.target_projection_head.load_state_dict(checkpoint["target_projection_head_state_dict"])
        self.current_step = checkpoint["current_step"]
        self.base_tau = checkpoint["base_tau"]
        self.fc = checkpoint["fc"]
        return checkpoint["optimiser_state_dict"]


    def create_fc(self, num_classes):
        if self.fc is not None:
            print("[Warning] fc layer is already instantiated")
        self.fc = torch.nn.Linear(in_features = self.embedding_size,
                                  out_features = num_classes)


    def get_all_online_params(self):
        parameters = []
        for model in (self.online_encoder, self.online_projection_head, self.online_predictor):
            parameters += list(model.parameters())
        return parameters


    @ staticmethod
    def regression_loss(predicted: torch.Tensor,
                        expected: torch.Tensor):
        return 2 - (2 * torch.nn.functional.cosine_similarity(x1 = predicted,
                                                              x2 = expected,
                                                              dim=-1))




    def forward(self,
                image_1,
                image_2 = None):

        ### Inference
        if image_2 is None:
            if self.fc is None:
                raise Exception("Output FC layer has not been initialised/loaded")
            return self.fc(self.online_encoder(image_1))
        online_output_1 = self.online_predictor(self.online_projection_head(self.online_encoder(image_1)))
        online_output_2 = self.online_predictor(self.online_projection_head(self.online_encoder(image_2)))
        with torch.no_grad():
            target_output_1 = self.target_projection_head(self.target_encoder(image_1))
            target_output_2 = self.target_projection_head(self.target_encoder(image_2))

        total_loss = self.regression_loss(online_output_1,target_output_2.detach()) + self.regression_loss(online_output_2,target_output_1.detach())
        return total_loss.mean()


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
        self.activation = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(num_features = size_out)
        self.linear = torch.nn.Linear(in_features = size_in,
                                      out_features = size_out,
                                      bias = bias)
        self.all_layers = torch.nn.Sequential(self.linear, self.batch_norm, self.activation)

    def forward(self,
                input_vector):
       return self.all_layers(input_vector)


class MLP(torch.nn.Module):

    def __init__(self,
                 num_hidden_layers: int,
                 size_in: int,
                 size_out: int,
                 hidden_layer_size : int,
                 batch_norm_enabled: bool):
        super(MLP,
              self).__init__()
        self.full_linear_block = LinearLayer(size_in = size_in,
                                             size_out = hidden_layer_size,
                                             activation = "relu",
                                             batch_norm_enabled = batch_norm_enabled)
        self.output_layer = torch.nn.Linear(in_features = hidden_layer_size,
                                            out_features = size_out)

        self.all_layers = torch.nn.Sequential(self.full_linear_block, self.output_layer)

    def forward(self,
                input_vector):
        return self.all_layers(input_vector)
