import copy
import math
import os

import torch.nn
from torchvision.models import get_model


# Possible activation functions for MLP
ACTIVATION_OPTIONS = {
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "none": torch.nn.Identity
}


class BYOL(torch.nn.Module):
    """ Class implementing the Boostrap-Your-Own-Latent (BYOL) architecture

    Full holds all appropriate training params such as ema tau and current step
    Holds the online and target network as well as the predictor but seperately
    If only one image is provided to the forward method is treated as inference.

    Attributes:
        name:
            Name of model when saving if not overridden
        fc:
            The output linear layer for use after the online_encoder
            Is None until either manually assigned or create_fc is called
        base_tau:
            The base tau for exponential moving average
        current_tau:
            The current tau value for exponential moving average
        current_step:
            The current step. Meant to increase from 1->max_num_steps
            Used to calculate exponential moving average tau
            Is incremented everytime target network is updated
        max_num_steps:
            The maximum number of steps
            Used when calculating exponential moving average tau
        embedding_size:
            Size output of the encoder networks
            Is determined by backbone output (ignoring any final linear layers)
        projection_size:
            Size output of projection layer
        input_height:
            Input height of model
        input_width:
            Input width of model
        online_encoder:
            Pytorch network representing the online encoder
        online_projection_head:
            Pytorch network representing the projection head
        online_predictor:
            Pytorch network representing online predictor
        target_encoder:
            Pytorch network representing target encoder
        target_projection_head:
            Pytorch network representing target projection head

    """

    def __init__(self,
                 max_num_steps : int,
                 encoder_model: str = "resnet50",
                 num_projection_layers: int = 1,
                 projection_size: int = 128,
                 num_predictor_layers: int = 1,
                 input_height: int = 224,
                 input_width: int = 224,
                 projection_hidden_layer_size: int = 128,
                 base_ema_tau: int = 0.996,
                 name: str = "byol_model"):
        """Initialiser method for BYOL network

        Args:
            max_num_steps:
                Max number of bckprops steps
                Used for calculation of exponential moving average tau value
            encoder_model:
                The backbone model. E.g. "Resnet50" or "Resnet18"
            num_projection_layers:
                Num hidden layers in projection head
                Currently not used
            projection_size:
                Output size of projection head network
            num_predictor_layers:
                Num hidden layers in predictor
                Currently not used
            input_height:
                Input height of model
            input_width:
                Input width of model
            projection_hidden_layer_size:
                Size of hidden layer in projection head/predictor
            base_ema_tau:
                Base tau value used in calculating exponential moving average tau value
            name:
                Name of model
                Used when saving the model
        """
        super().__init__()
        self.name = name
        self.fc = None
        self.base_tau = base_ema_tau
        self.current_tau = base_ema_tau
        self.current_step = 0
        self.max_num_steps = max_num_steps
        self.embedding_size = embedding_size
        self.projection_size = projection_size
        self.input_height = input_height
        self.input_width = input_width

        self.online_encoder = get_model(name=encoder_model,
                                        weights=None)
        self.embedding_size = self.online_encoder.fc.in_features
        # Remove the linear output layer
        self.online_encoder.fc = torch.nn.Identity()

        # Create online projection head
        self.online_projection_head = MLP(num_hidden_layers=num_projection_layers,
                                          size_in=self.embedding_size,
                                          hidden_layer_size=projection_hidden_layer_size,
                                          size_out=self.projection_size,
                                          batch_norm_enabled=True)

        # Create online predictor
        self.online_predictor = MLP(num_hidden_layers=num_projection_layers,
                                    size_in=self.projection_size,
                                    size_out=self.projection_size,
                                    hidden_layer_size=projection_hidden_layer_size,
                                    batch_norm_enabled=True)

        # Create target network
        self.target_encoder, self.target_projection_head = self.create_target_network()


    def create_target_network(self):
        """Creates target network with gradient descent disabled

        It does this by creating deep copies of the encoder and projection head
        Also sets the requires_grad param for all layers to False so target network is not
        effected by gradient descent

        Returns:

        """
        target_encoder = copy.deepcopy(self.online_encoder)
        target_projection_head = copy.deepcopy(self.online_projection_head)

        # Prevent gradients being calculated/updated by grad descent
        for param in list(target_encoder.parameters()) + list(target_projection_head.parameters()):
            param.requires_grad = False
        return target_encoder, target_projection_head

    def _update_tau(self):
        """Method to update current_tau based on current and max steps

        Returns:
            None
        """
        assert self.max_num_steps is not None, "For EMA the max number of training steps must be specified"
        #  New Tau = 1 - (1-base_tau) * (cos(pi * k/K)+1)/2
        # As in BYOL paper
        self.current_tau = 1 - (1 - self.base_tau) * (
                math.cos((math.pi * self.current_step) / self.max_num_steps) + 1) / 2

    def set_max_num_steps(self,
                          new_max_num_steps : int):
        """ Setter for max_num_steps attribute

        Args:
            new_max_num_steps:
                The new value to assign to max_num_steps

        Returns:
            None

        """
        self.max_num_steps = new_max_num_steps

    def update_target_network(self):
        """ Method to update target network and tau. Expected to be called every backprop

        Returns:
            None
        """
        self.current_step += 1
        self._update_tau()
        with torch.no_grad():
            for online_encoder_param, target_encoder_param in zip(self.online_encoder.parameters(),
                                                                  self.target_encoder.parameters()):
                target_encoder_param.data = target_encoder_param.data * self.current_tau + (
                        1 - self.current_tau) * online_encoder_param.data
            for online_projector_param, target_projector_param in zip(self.online_projection_head.parameters(),
                                                                      self.target_projection_head.parameters()):
                target_projector_param.data = target_projector_param.data * self.current_tau + (
                        1 - self.current_tau) * online_projector_param.data

    def save(self,
             folder_path,
             epoch,
             optimiser,
             model_save_name=None):
        """Method to save BYOL model

        Saves model to folder_path/model_name_epoch={epoch}.pt if model_save_name is None
        Else folder_path/model_save_name if model_save_name is not None

        Args:
            folder_path:
                Path to folder model will be saved within
                Not the full path incl the model
                E.g. /path/to/folder
            epoch:
                Current epoch, used in the name of the saved model when not overriden by model_save_name
                Also saved within the model dict
            optimiser:
                Optimiser being used to train model. Optimiser state dict will be saved with mode dict
                for use if resuming training
            model_save_name:
                Defaults to None. Is used to override default file name when saving

        Returns:

        """
        if model_save_name is None:
            model_save_name = f"{self.name}_epoch={epoch}.pt"
        save_path = os.path.join(folder_path,
                                 model_save_name)
        print(f"Saving model to {save_path}")
        torch.save({
            "epoch": epoch,
            "online_encoder_state_dict": self.online_encoder.state_dict(),
            "online_projection_head_state_dict": self.online_projection_head.state_dict(),
            "online_predictor_state_dict": self.online_predictor.state_dict(),
            "target_encoder_state_dict": self.target_encoder.state_dict(),
            "target_projection_head_state_dict": self.target_projection_head.state_dict(),
            "optimiser_state_dict": optimiser.state_dict(),
            "current_step": self.current_step,
            "base_tau": self.base_tau,
            "fc": self.fc

            }, save_path)
        return save_path

    def load(self, model_path):
        """ Load function

        Will load all networks, step, tau and final output linear layer

        Args:
            model_path:
                Path to model to load

        Returns:
            optimiser state dictioniary of saved model
        """
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

    def create_fc(self,
                  num_classes):
        """ Creates output linear layer for classification and assigns it to BYOL fc attribute
        Args:
            num_classes:
                Num outputs of classification layer, equal to num classes
        Returns:
            None
        """
        if self.fc is not None:
            print("[Warning] fc layer is already instantiated")
        self.fc = torch.nn.Linear(in_features=self.embedding_size,
                                  out_features=num_classes)


    @staticmethod
    def regression_loss(predicted: torch.Tensor,
                        expected: torch.Tensor):
        """ Static regression loss function used by BYOL

        Args:
            predicted:
                Output of predictor
            expected:
                Output of target network

        Returns:
            float : loss
        """
        return 2 - (2 * torch.nn.functional.cosine_similarity(x1=predicted,
                                                              x2=expected,
                                                              dim=-1))

    def forward(self,
                image_1 : torch.Tensor,
                image_2 : torch.Tensor=None):
        """Forward step

        In case only one image is provided for param image_1 then inference is done, producing classification output.
        In case both images are provided, then regression loss is calculated averaged and returned

        Args:
            image_1:
                View 1 of image
            image_2:
                Defaults to None
                View 2 of image if self-supervised pre-training is done


        Returns:
            If one image is passed
                torch.Tensor : Classification output
            If both image views are passed
                torch.Tensor : Average regression loss
        """

        if image_1.shape[-2:] != (self.input_height, self.input_width):
            raise ValueError(f"Expected image shape of {(self.input_height, self.input_width)} but got {image_2.shape[-2:]} for image_1")
        ### Inference
        if image_2 is None:
            if self.fc is None:
                raise Exception("Output FC layer has not been initialised/loaded")
            return self.fc(self.online_encoder(image_1))

        if image_2.shape[-2:] != (self.input_height, self.input_width):
            raise ValueError(f"Expected image shape of {(self.input_height, self.input_width)} but got {image_2.shape[-2:]} for image_1")

        online_output_1 = self.online_predictor(self.online_projection_head(self.online_encoder(image_1)))
        online_output_2 = self.online_predictor(self.online_projection_head(self.online_encoder(image_2)))

        # Target network does not need to be updated
        with torch.no_grad():
            target_output_1 = self.target_projection_head(self.target_encoder(image_1))
            target_output_2 = self.target_projection_head(self.target_encoder(image_2))

        total_loss = self.regression_loss(online_output_1,
                                          target_output_2.detach()) + self.regression_loss(
            online_output_2,
            target_output_1.detach())
        return total_loss.mean()


class LinearLayer(torch.nn.Module):
    """A Linear layer class implementing a linear layer, batch norm and relu activation

    Attributes:
        activation:
            torch relu activation function
        batch_norm:
            torch batch norm
        linear:
            torch Linear Layer
        all_layers:
            sequential list of layers

    """

    def __init__(self,
                 size_in: int,
                 size_out: int,
                 bias: bool = True,
                 batch_norm_enabled: bool = True,
                 activation="relu"):
        """ Init method, declares all individual layers/torch.nn objects

        Args:
            size_in:
                Input size for layer
            size_out:
                Output size returned by layer
            bias:
                Whether bias is enabled
            batch_norm_enabled:
                Whether batch norm is enabled
                Not currently in use
            activation:
                Activation type
                Not currently in use
        """
        super().__init__()

        if activation not in ACTIVATION_OPTIONS:
            raise ValueError(f"Expected activation function to be one of {list(ACTIVATION_OPTIONS.keys())}. Received"
                             f" {activation}")
        self.activation = torch.nn.ReLU()
        self.batch_norm = torch.nn.BatchNorm1d(num_features=size_out)
        self.linear = torch.nn.Linear(in_features=size_in,
                                      out_features=size_out,
                                      bias=bias)
        self.all_layers = torch.nn.Sequential(self.linear,
                                              self.batch_norm,
                                              self.activation)

    def forward(self,
                input_vector : torch.Tensor):
        """Forward method for layer

        Args:
            input_vector:
                Input array

        Returns:
            torch.Tensor : Layer output
        """
        return self.all_layers(input_vector)


class MLP(torch.nn.Module):
    """Multi-Layer Perceptron class

    Attributes:
        full_linear_block:
            Input linear layer
        output_layer :
            Output Linear Layer
        all_layers:
            torch.nn.Sequential : Sequential list of layers
    """

    def __init__(self,
                 num_hidden_layers: int,
                 size_in: int,
                 size_out: int,
                 hidden_layer_size: int,
                 batch_norm_enabled: bool):
        """ Init method

        Args:
            num_hidden_layers:
                Num hidden layers (Total num layers = num_hidden_layers - 1)
            size_in:
                Input size of MLP
            size_out:
                Output size of MLP
            hidden_layer_size:
                Size of hidden layers
            batch_norm_enabled:
                Whether batch norm is enabled
        """
        super(MLP,
              self).__init__()
        self.full_linear_block = LinearLayer(size_in=size_in,
                                             size_out=hidden_layer_size,
                                             activation="relu",
                                             batch_norm_enabled=batch_norm_enabled)
        self.output_layer = torch.nn.Linear(in_features=hidden_layer_size,
                                            out_features=size_out)

        self.all_layers = torch.nn.Sequential(self.full_linear_block,
                                              self.output_layer)

    def forward(self,
                input_vector : torch.Tensor):
        """Forward method

        Args:
            input_vector:
                Input array

        Returns:
            torch.Tensor : MLP output
        """
        return self.all_layers(input_vector)
