import random
from typing import Tuple

import PIL.Image
import torch
import torchvision.transforms
from torch.distributions import Uniform
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms.functional import adjust_hue, adjust_brightness, adjust_contrast, adjust_saturation, \
    InterpolationMode


class BYOLAugmenter:
    """ Class for applying augmentations used in BYOL paper

    Attributes:
        resize_output_height : An integer, will determine the output height of any Resize/ResizeCrops transforms
        resize_output_width : An integer, will determine the output width of any Resize/ResizeCrops transforms
    """

    def __init__(self,
                 resize_output_height: int,
                 resize_output_width: int):
        """ Inits BYOL augmenter class

        Args:
            resize_output_height : An integer, will determine the output height of any Resize/ResizeCrops transforms
            resize_output_width : An integer, will determine the output width of any Resize/ResizeCrops transforms
        """
        self.resize_output_height = resize_output_height
        self.resize_output_width = resize_output_width
        self.view_1 = None
        self.view_2 = None

    def self_supervised_pre_train_transform(self,
                                            image: torch.Tensor):
        """Method for returning two views of the image

        Requires setup_multi_view method to be called first
        Args:
            image : Image to be augmented into two different views

        Returns:
            torch.Tensor, Torch.Tensor : Two differing augmented tensors, representing the two views
        Raises:
            Exception : An error occurs if view_1 and view_2 transform compositions have not been setup
        """
        if None in (self.view_1, self.view_2):
            raise Exception("setup_multi_view method must be called prior to pre train transform being used")
        return self.view_1(image), self.view_2(image)

    def setup_multi_view(self,
                         view_1_params: dict,
                         view_2_params: dict):
        """Method for setting up the view transforms based on the provide params

        Assigns view_1 and view_2 attributes

        Args:
            view_1_params:
                Parameters to compose view 1 augmentations
                Should be a dict with keys being augmentations types and values being kwargs for that transforms params
            view_2_params:
                Parameters to compose view 2 augmentations
                Should be a dict with keys being augmentations types and values being kwargs for that transforms params

        Returns:
            None

        """
        self.view_1 = self.create_view(**view_1_params)
        self.view_2 = self.create_view(**view_2_params)

    def get_fine_tune_augmentations(self,
                                    resize_crop: dict,
                                    random_flip: dict,
                                    normalise: dict):
        """ Method to get fine tune augmentations composition

        Args:
            resize_crop:
                Dictionary containing kargs for BYOLRandomResize excluding output_height and output_width
            random_flip:
                Dictionary containing kargs for BYOLHorizontalFlip
            normalise:
                Dictionary containing kargs for torchvision Normalize

        Returns:
            torchvision transform composition consisting of
            BYOLRandomResize, BYOLHorizontalFlip, ToTensor and Normalize
        """
        return torchvision.transforms.Compose([BYOLRandomResize(output_height=self.resize_output_height,
                                                                output_width=self.resize_output_width,
                                                                **resize_crop),
                                               torchvision.transforms.Resize(size = (self.resize_output_height,
                                                                                     self.resize_output_width),
                                                                             interpolation = InterpolationMode.BICUBIC),
                                               BYOLHorizontalFlip(**random_flip),
                                               ToTensor(),
                                               Normalize(**normalise)])

    def get_test_augmentations(self,
                               normalise: dict):
        """Method to get test augmentations composition

        Args:
            normalise:
                Dictionary containing kargs for torchvision Normalize

        Returns:
            torchvision transform composition consisting of
            Resize, ToTensor and Normalize

        """
        return torchvision.transforms.Compose([torchvision.transforms.Resize(size=(self.resize_output_height,
                                                                                   self.resize_output_width),
                                                                             interpolation=InterpolationMode.BICUBIC),
                                               ToTensor(),
                                               Normalize(**normalise)])

    def create_view(self,
                    resize_crop : dict,
                    random_flip_vertical : dict,
                    colour_jitter : dict,
                    gaussian_blur : dict,
                    solarize : dict,
                    normalise : dict,
                    colour_drop : dict,
                    random_flip_horizontal : dict,
                    random_affine : dict,
                    random_perspective : dict):
        """ Method to create torchvision transform compositions for creating BYOL views

        Args:
            resize_crop:
                Dictionary containing kargs for BYOLRandomResize excluding output_height and output_width
            random_flip:
                Dictionary containing kargs for BYOLHorizontalFlip
            colour_jitter:
                Dictionary containing kargs for BYOLRandomColorJitter
            gaussian_blur:
                Dictionary containing kargs for BYOLGaussianBlur
            solarize:
                Dictionary containing kargs for BYOLSolarize
            normalise:
                Dictionary containing kargs for torchvision Normalize
            colour_drop:
                Dictionary containing kargs for BYOLColorDrop

        Returns:
            torchvision transforms composition consisting of
            BYOLRandomResize, BYOLHorizontalFlip, BYOLRandomColorJitter, BYOLColourDrop,
            BYOLGaussianBlur, BYOLSolarize, torchvision ToTensor and torchvision Normalize
        """
        view_augs = []
        view_augs.append(BYOLRandomResize(output_height=self.resize_output_height,
                                          output_width=self.resize_output_width,
                                          **resize_crop))
        view_augs.append(torchvision.transforms.Resize(size = (self.resize_output_height,
                                                               self.resize_output_width),
                                                       interpolation = InterpolationMode.BICUBIC))
        view_augs.append(BYOLRandomAffine(**random_affine))
        view_augs.append(BYOLRandomPerspective(**random_perspective))
        view_augs.append(BYOLHorizontalFlip(**random_flip_horizontal))
        view_augs.append(BYOLVerticalFlip(**random_flip_vertical))
        view_augs.append(BYOLRandomColourJitter(**colour_jitter))
        view_augs.append(BYOLColourDrop(**colour_drop))
        view_augs.append(BYOLGaussianBlur(**gaussian_blur))
        view_augs.append(BYOLSolarize(**solarize))
        view_augs.append(ToTensor())
        view_augs.append(Normalize(**normalise))

        return torchvision.transforms.Compose(view_augs)



class BYOLRandomAffine(BYOLRandomApplyAug):

    def __init__(self,
                 apply_probability : float,
                 degrees : int,
                 translate : Tuple[int, int] = None,
                 scale : Tuple[int, int] = None,
                 shear : int = None):
        """Init method
        Args:
            apply_probability:
                Min probability to apply random affine
        """
        super().__init__(apply_probability)
        self.aug = torchvision.transforms.RandomAffine(degrees=degrees,
                                                       translate=translate,
                                                       scale=scale,
                                                       shear=shear,
                                                       interpolation=InterpolationMode.BICUBIC)

class BYOLRandomPerspective(BYOLRandomApplyAug):

    def __init__(self,
                 apply_probability : float,
                 distortion_scale : float):
        """Init method
        Args:
            apply_probability:
                Min probability to apply random affine
        """
        super().__init__(apply_probability)
        self.aug = torchvision.transforms.RandomPerspective(distortion_scale=distortion_scale,
                                                            interpolation=InterpolationMode.BICUBIC)
class BYOLRandomApplyAug:
    """Abstract class used to make any augmentation be able to be randomly applied

    Attributes:
        apply_probability:
            The chance the augmentation assigned to the aug attribute will be applied to the input image
        aug:
            The augmentation that may be applied to the input image

    """
    def __init__(self,
                 apply_probability : float):
        """Initialises abstract class with None for aug

        Args:
            apply_probability:
                float between 0 and 1.0 representing the chance the augmentation will be applied to the image
        """
        self.apply_probability = apply_probability
        self.aug = None

    def __call__(self,
                 image : torch.Tensor):
        """Method to apply augment

        Args:
            image:
                The image to be augmented

        Returns:
            Post augmented image if prob less than min apply probability else original image
        """
        if random.random() < self.apply_probability:
            image = self.aug(image)

        return image


class BYOLRandomResize(BYOLRandomApplyAug):
    """Class to augment an image randomly using RandomResize torchvision augment

    Attributes:
        aug:
            Torchvision transforms RandomResizedCrop instance
    """

    def __init__(self,
                 apply_probability : float,
                 output_height : int,
                 output_width : int):
        """Init method

        Args:
            apply_probability:
                Min probability to apply RandomResizedCrop transform to image
            output_height:
                Output height post random crop
            output_width:
                Output width post random crop
        """
        super().__init__(apply_probability)
        self.aug = torchvision.transforms.RandomResizedCrop(size=(output_height, output_width),
                                                            interpolation=InterpolationMode.BICUBIC)


class BYOLHorizontalFlip(BYOLRandomApplyAug):
    """Class to augment an image randomly using horizontal flip torchvision augment

        Attributes:
            aug:
                Torchvision transforms functional hflip
    """

    def __init__(self,
                 apply_probability : float):
        """Init method
        Args:
            apply_probability:
                Min probability to apply horizontal flip
        """
        super().__init__(apply_probability)
        self.aug = torchvision.transforms.functional.hflip

class BYOLVerticalFlip(BYOLRandomApplyAug):
    """Class to augment an image randomly using horizontal flip torchvision augment

        Attributes:
            aug:
                Torchvision transforms functional hflip
    """

    def __init__(self,
                 apply_probability : float):
        """Init method
        Args:
            apply_probability:
                Min probability to apply horizontal flip
        """
        super().__init__(apply_probability)
        self.aug = torchvision.transforms.functional.vflip


class BYOLRandomColourJitter(BYOLRandomApplyAug):
    """Class to augment an image randomly using ColourJitter torchvision augment

        Attributes:
            aug:
                Torchvision transforms ColourJitter instance
    """

    def __init__(self,
                 apply_probability : float,
                 brightness_delta : float,
                 contrast_delta : float,
                 saturation_delta : float,
                 hue_delta : float):
        """Init method

        Args:
            apply_probability:
                Min probability to apply ColourJitter
            brightness_delta:
                The max possible brightness change
                Between 0.0 and 1.0
            contrast_delta:
                The max possible contrast change
                Between 0.0 and 1.0
            saturation_delta:
                The max possible saturation change
                Between 0.0 and 1.0
            hue_delta:
                The max possible hue change
                Betwen 0.0 and 0.5
        """
        super().__init__(apply_probability)
        self.aug = BYOLColourJitter(brightness_delta=brightness_delta,
                                    contrast_delta=contrast_delta,
                                    saturation_delta=saturation_delta,
                                    hue_delta=hue_delta)


class BYOLGaussianBlur(BYOLRandomApplyAug):
    """Class to augment an image randomly using GaussianBlur torchvision augment

        Attributes:
            aug:
                Torchvision transforms GaussianBlur instance
    """

    def __init__(self,
                 apply_probability,
                 kernel_size,
                 sigma):
        """Init method

        Args:
            apply_probability:
                Min probability to apply GaussianBlur augment
            kernel_size:
                Kernelsize for Gaussian Blur
            sigma:
                Sigma for Gaussian Blur
        """
        super().__init__(apply_probability)
        self.aug = torchvision.transforms.GaussianBlur(kernel_size=kernel_size,
                                                       sigma=sigma)


class BYOLSolarize(BYOLRandomApplyAug):
    """Class to augment an image randomly using Solarize torchvision augment

        Attributes:
            aug:
                Torchvision transforms functional solarize
            threshold:
                Threshold for solarization to apply

    """

    def __init__(self,
                 apply_probability,
                 threshold):
        super().__init__(apply_probability)
        self.aug = torchvision.transforms.functional.solarize
        self.threshold = threshold

    def __call__(self,
                 image : torch.Tensor):
        """Applys solarize condition based on apply_probability
        Args:
            image:
                Image to be augmented using solarize torchvision transform, based on the

        Returns:
            torch.Tensor of either solarized image or unaugmented (original) image
        """
        if random.random() < self.apply_probability:
            image = self.aug(image,
                             self.threshold)
        return image


class BYOLColourJitter():
    """ Class for performing ColorJitter, performing the adjustments in random order

    Difference between this and torchvision ColorJitter is that the individual components
    representing the jitter are performed in a random order

    Attributes:
        colour_jitter_components (List[tuple(transform, float)]):
            List of transforms to be applied in random order and their corresponding max delta



    """

    def __init__(self,
                 brightness_delta : float =0.5,
                 contrast_delta : float=0.5,
                 saturation_delta : float =0.2,
                 hue_delta : float =0.1):
        self.colour_jitter_components = [(adjust_hue, hue_delta),
                                         (adjust_brightness, brightness_delta),
                                         (adjust_contrast, contrast_delta),
                                         (adjust_saturation, saturation_delta)]

    def __call__(self,
                 image : torch.Tensor):
        """ Method to performs colour jitter on image in a random order

        Uses the max delta to determine the min/max values for the
        functional transforms. Randomises the order the transforms are done

        Args:
            image:
                Image to be augmented

        Returns:
            torch.Tensor :
                Colour jittered image dependent upon probability else unaugmented image
        """
        random.shuffle(self.colour_jitter_components)
        for component, adjustment_max_value in self.colour_jitter_components:
            # No change for hue is around 0 vs 1 for all other colour jitter transforms
            if component == adjust_hue:
                no_change_val = 0
            else:
                no_change_val = 1
            adjustment_val = Uniform(no_change_val - adjustment_max_value,
                                     no_change_val + adjustment_max_value).sample()
            image = component(image,
                              adjustment_val)

        return image


class BYOLColourDrop(torchvision.transforms.RandomGrayscale):
    """Method to perform colour drop (grayscale conversion) according to the BYOL paper

    Attributes:
        aug :
         None
        apply_probability:
            min float to apply augmentation
    """
    def __init__(self,
                 apply_probability: float):
        """

        Args:
            apply_probability:
                Min probability to apply grayscale conversion
        """
        super().__init__(p=apply_probability)

