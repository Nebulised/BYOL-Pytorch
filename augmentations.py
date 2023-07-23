import math
import random
from typing import Tuple, Optional, List

import PIL.Image
import numpy as np
import torch
import torchvision.transforms
from matplotlib import pyplot as plt
from torch import Tensor
from torch.distributions import Uniform
from torchvision.transforms import ToTensor, Normalize
from torchvision.transforms.functional import adjust_hue, adjust_brightness, adjust_contrast, adjust_saturation, \
    InterpolationMode
from torchvision.transforms import RandomApply


class GreyscaleToRGB():

    def __call__(self, image):
        assert image.size()[0] == 1
        return image.repeat(3,1,1)
class ScaleTensor:

    def __call__(self, image):
        return image.to(dtype=torch.get_default_dtype()).div(255)


class BYOLRandomApplyAug:
    """Abstract class used to make any augmentation be able to be randomly applied

    Attributes:
        apply_probability:
            The chance the augmentation assigned to the aug attribute will be applied to the input image
        aug:
            The augmentation that may be applied to the input image

    """

    def __init__(self,
                 apply_probability: float,
                 aug: torchvision.transforms):
        """Initialises abstract class with None for aug

        Args:
            apply_probability:
                float between 0 and 1.0 representing the chance the augmentation will be applied to the image
        """
        self.aug = torchvision.transforms.RandomApply(p = apply_probability,
                                                      transforms = [aug])


    def __call__(self,
                 image: torch.Tensor):
        """Method to apply augment

        Args:
            image:
                The image to be augmented

        Returns:
            Post augmented image if prob less than min apply probability else original image
        """
        return self.aug(image)


class CustomAugApplicator:

    def __init__(self,
                 augmentation,
                 apply_both_if_applied: bool,
                 apply_probability :float,
                 duplicate_aug: bool,
                 **params):
        self.apply_both_if_applied = apply_both_if_applied
        self.duplicate_augmentation = duplicate_aug
        if self.apply_both_if_applied:
            self.apply_probability = apply_probability

            self.aug = augmentation(apply_probability = 1.0,
                                    **params)
        else:
            self.aug = augmentation(apply_probability = apply_probability,
                                    **params)

    def __call__(self,
                 image_view_1,
                 image_view_2):
        if self.apply_both_if_applied:
            if torch.rand(1) < self.apply_probability:
                if self.duplicate_augmentation:
                    stacked_images = torch.stack(image_view_1.unsqueeze(0), image_view_2.unsqueeze(0))
                    return self.aug(stacked_images)
                else:
                    return self.aug(image_view_1), self.aug(image_view_2)
            else:
                return image_view_1, image_view_2
        else:
            return self.aug(image_view_1), self.aug(image_view_2)


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
        self.custom_aug_list = None
        self._input_is_grayscale = None

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
        if len(normalise["mean"]) == 3:
            is_grayscale = False
        else:
            is_grayscale = True

        fine_tune_aug_list = [torchvision.transforms.PILToTensor()]
        if is_grayscale:
            fine_tune_aug_list.append(torchvision.transforms.Grayscale(num_output_channels=1))
        fine_tune_aug_list += [BYOLRandomResize(output_height = self.resize_output_height,
                                                output_width = self.resize_output_width,
                                                **resize_crop),
                               torchvision.transforms.Resize(size = (self.resize_output_height,
                                                                     self.resize_output_width),
                                                             interpolation = InterpolationMode.BICUBIC),
                               BYOLHorizontalFlip(**random_flip),
                               ScaleTensor(),
                               Normalize(**normalise)]
        if is_grayscale:
            fine_tune_aug_list.append(GreyscaleToRGB())
        return torchvision.transforms.Compose(fine_tune_aug_list)

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
        if len(normalise["mean"]) == 3:
            is_grayscale = False
        else:
            is_grayscale = True

        test_aug_list = [torchvision.transforms.PILToTensor()]
        if is_grayscale:
            test_aug_list.append(torchvision.transforms.Grayscale(num_output_channels=1))
        test_aug_list += [torchvision.transforms.Resize(size = (self.resize_output_height,
                                                                self.resize_output_width),
                                                        interpolation = InterpolationMode.BICUBIC),
                               ScaleTensor(),
                               Normalize(**normalise)]
        if is_grayscale:
            test_aug_list.append(GreyscaleToRGB())
        return torchvision.transforms.Compose(test_aug_list)

    def setup_custom_view(self,
                          resize_crop: dict,
                          random_flip_vertical: dict,
                          colour_jitter: dict,
                          gaussian_blur: dict,
                          solarize: dict,
                          normalise: dict,
                          colour_drop: dict,
                          random_flip_horizontal: dict,
                          random_affine: dict,
                          random_perspective: dict,
                          cut_paste: dict,
                          cut_paste_scar: dict,
                          cut_paste_affine: dict):
        self.custom_aug_list = []
        resize_crop["output_height"] = self.resize_output_height
        resize_crop["output_width"] = self.resize_output_width
        self.custom_aug_list.append(CustomAugApplicator(BYOLRandomResize,
                                                        **resize_crop))
        self.custom_aug_list.append(CustomAugApplicator(BYOLRandomAffine,
                                                        **random_affine))
        self.custom_aug_list.append(torchvision.transforms.Resize(size = (self.resize_output_height,
                                                                          self.resize_output_width),
                                                                  interpolation = InterpolationMode.BICUBIC))
        self.custom_aug_list.append(CustomAugApplicator(BYOLHorizontalFlip,
                                                        **random_flip_horizontal))
        self.custom_aug_list.append(CustomAugApplicator(BYOLVerticalFlip,
                                                        **random_flip_vertical))
        self.custom_aug_list.append(CustomAugApplicator(BYOLCutPaste,
                                                        **cut_paste))
        self.custom_aug_list.append(CustomAugApplicator(BYOLCutPasteScar,
                                                        **cut_paste_scar))

        self.custom_aug_list.append(CustomAugApplicator(BYOLCutPasteAffine,
                                                        **cut_paste_affine))
        self.custom_aug_list.append(CustomAugApplicator(BYOLRandomPerspective,
                                                        **random_perspective))
        self.custom_aug_list.append(CustomAugApplicator(BYOLRandomColourJitter,
                                                        **colour_jitter))
        self.custom_aug_list.append(CustomAugApplicator(BYOLColourDrop,
                                                        **colour_drop))
        self.custom_aug_list.append(CustomAugApplicator(BYOLGaussianBlur,
                                                        **gaussian_blur))
        self.custom_aug_list.append(CustomAugApplicator(BYOLSolarize,
                                                        **solarize))
        self.custom_aug_list.append(ScaleTensor())
        self.custom_aug_list.append(Normalize(**normalise))
        if len(normalise["mean"]) == 1:
            self._input_is_grayscale = True
            self.custom_aug_list.append(GreyscaleToRGB())
        else:
            self._input_is_grayscale = False

    def apply_custom_view(self,
                          image):
        image = torchvision.transforms.functional.pil_to_tensor(image)
        if self._input_is_grayscale is None:
            raise Exception("setup_custom_view must be called beforehand")
        if self._input_is_grayscale:
            image = torchvision.transforms.functional.rgb_to_grayscale(image, num_output_channels=1)
        image_view_1, image_view_2 = image.clone(), image.clone()
        for each_transform in self.custom_aug_list:
            if isinstance(each_transform, CustomAugApplicator):
                image_view_1, image_view_2 = each_transform(image_view_1,
                                                            image_view_2)
            else:
                image_view_1, image_view_2 = each_transform(image_view_1), each_transform(image_view_2)
        return image_view_1, image_view_2

    def create_view(self,
                    resize_crop: dict,
                    random_flip_vertical: dict,
                    colour_jitter: dict,
                    gaussian_blur: dict,
                    solarize: dict,
                    normalise: dict,
                    colour_drop: dict,
                    random_flip_horizontal: dict,
                    random_affine: dict,
                    random_perspective: dict,
                    cut_paste: dict,
                    cut_paste_scar: dict,
                    cut_paste_affine: dict):
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
        view_augs.append(BYOLRandomResize(output_height = self.resize_output_height,
                                          output_width = self.resize_output_width,
                                          **resize_crop))
        view_augs.append(BYOLRandomAffine(**random_affine))
        view_augs.append(torchvision.transforms.Resize(size = (self.resize_output_height,
                                                               self.resize_output_width),
                                                       interpolation = InterpolationMode.BICUBIC))
        view_augs.append(BYOLHorizontalFlip(**random_flip_horizontal))
        view_augs.append(BYOLVerticalFlip(**random_flip_vertical))
        view_augs.append(BYOLCutPaste(**cut_paste))
        view_augs.append(BYOLCutPasteScar(**cut_paste_scar))
        view_augs.append(BYOLCutPasteAffine(**cut_paste_affine))
        view_augs.append(BYOLRandomPerspective(**random_perspective))
        view_augs.append(BYOLRandomColourJitter(**colour_jitter))
        view_augs.append(BYOLColourDrop(**colour_drop))
        view_augs.append(BYOLGaussianBlur(**gaussian_blur))
        view_augs.append(BYOLSolarize(**solarize))
        view_augs.append(ToTensor())
        view_augs.append(Normalize(**normalise))

        return torchvision.transforms.Compose(view_augs)


class BYOLRandomResize(BYOLRandomApplyAug):
    """Class to augment an image randomly using RandomResize torchvision augment

    Attributes:
        aug:
            Torchvision transforms RandomResizedCrop instance
    """

    def __init__(self,
                 apply_probability: float,
                 output_height: int,
                 output_width: int):
        """Init method

        Args:
            apply_probability:
                Min probability to apply RandomResizedCrop transform to image
            output_height:
                Output height post random crop
            output_width:
                Output width post random crop
        """
        super().__init__(apply_probability,
                         aug = torchvision.transforms.RandomResizedCrop(size = (output_height, output_width),
                                                                        interpolation = InterpolationMode.BICUBIC))


class BYOLHorizontalFlip(torchvision.transforms.RandomHorizontalFlip):
    """Class to augment an image randomly using horizontal flip torchvision augment

        Attributes:
            aug:
                Torchvision transforms functional hflip
    """

    def __init__(self,
                 apply_probability: float):
        """Init method
        Args:
            apply_probability:
                Min probability to apply horizontal flip
        """
        super().__init__(p = apply_probability)


class BYOLVerticalFlip(torchvision.transforms.RandomVerticalFlip):
    """Class to augment an image randomly using vertical flip torchvision augment

        Attributes:
            aug:
                Torchvision transforms functional vflip
    """

    def __init__(self,
                 apply_probability: float):
        """Init method
        Args:
            apply_probability:
                Min probability to apply vertical flip
        """
        super().__init__(p = apply_probability)


class BYOLRandomColourJitter(BYOLRandomApplyAug):
    """Class to augment an image randomly using ColourJitter torchvision augment

        Attributes:
            aug:
                Torchvision transforms ColourJitter instance
    """

    def __init__(self,
                 apply_probability: float,
                 brightness_delta: float,
                 contrast_delta: float,
                 saturation_delta: float,
                 hue_delta: float):
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
        super().__init__(apply_probability,
                         aug = torchvision.transforms.ColorJitter(brightness = brightness_delta,
                                                                  contrast = contrast_delta,
                                                                  saturation = saturation_delta,
                                                                  hue = hue_delta))


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
        super().__init__(apply_probability,
                         aug = torchvision.transforms.GaussianBlur(kernel_size = kernel_size,
                                                                   sigma = sigma))


class BYOLSolarize(torchvision.transforms.RandomSolarize):
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
        super().__init__(p = apply_probability,
                         threshold = threshold)


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
        super().__init__(p = apply_probability)


class BYOLRandomAffine(BYOLRandomApplyAug):

    def __init__(self,
                 apply_probability: float,
                 degrees: int,
                 translate: Tuple[int, int] = None,
                 scale: Tuple[int, int] = None,
                 shear: int = None):
        """Init method
        Args:
            apply_probability:
                Min probability to apply random affine
        """
        super().__init__(apply_probability,
                         aug = torchvision.transforms.RandomAffine(degrees = degrees,
                                                                   translate = translate,
                                                                   scale = scale,
                                                                   shear = shear,
                                                                   interpolation = InterpolationMode.BILINEAR))


class BYOLRandomPerspective(BYOLRandomApplyAug):

    def __init__(self,
                 apply_probability: float,
                 distortion_scale: float):
        """Init method
        Args:
            apply_probability:
                Min probability to apply random affine
        """
        super().__init__(apply_probability,
                         aug = torchvision.transforms.RandomPerspective(distortion_scale = distortion_scale,
                                                                        interpolation = InterpolationMode.BILINEAR))


class BYOLCutPaste(torchvision.transforms.RandomErasing):

    def __init__(self,
                 apply_probability: float,
                 scale = (0.02, 0.15),
                 ratio = (0.3, 1),
                 colour_jitter_level: float = 0.1):
        super(BYOLCutPaste,
              self).__init__(p = apply_probability,
                             scale = scale,
                             ratio = ratio,
                             value = 255,
                             inplace = False)

        self.aug = torchvision.transforms.Compose([torchvision.transforms.ColorJitter(brightness = colour_jitter_level,
                                                                                      contrast = colour_jitter_level,
                                                                                      saturation = colour_jitter_level,
                                                                                      hue = colour_jitter_level)])

    def forward(self,
                img):
        if torch.rand(1) < self.p:

            # cast self.value to script acceptable type
            if isinstance(self.value,
                          (int, float)):
                value = [float(self.value)]
            elif isinstance(self.value,
                            str):
                value = None
            elif isinstance(self.value,
                            (list, tuple)):
                value = [float(v) for v in self.value]
            else:
                value = self.value
            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                        "If value is a sequence, it should have either a single value or "
                        f"{img.shape[-3]} (number of input channels)"
                        )

            img_h, img_w = img.shape[-2], img.shape[-1]
            i, j, h, w, v = self.get_params(img,
                                            scale = self.scale,
                                            ratio = self.ratio,
                                            value = value)
            if h == img_h or w == img_w:
                print("Warning, did not find a solution for cutpaste. Skipping")
                return img
            destination_i = torch.randint(0,
                                          img_h - h + 1,
                                          size = (1,)).item()
            destination_j = torch.randint(0,
                                          img_w - w + 1,
                                          size = (1,)).item()

            aug_img = self.aug(img[..., i: i + h, j: j + w])
            img = img.clone()
            img[..., destination_i: destination_i + h, destination_j: destination_j + w][aug_img != v] = aug_img[aug_img != v]

        return img


class BYOLCutPasteScar(BYOLCutPaste):

    def __init__(self,
                 apply_probability: float,
                 colour_jitter_level: float = 0.1,
                 degrees : int = 45):
        super().__init__(apply_probability=apply_probability,
                         scale = (0, 0),
                         ratio = (0, 0),
                         colour_jitter_level = colour_jitter_level)
        self.aug = torchvision.transforms.Compose([torchvision.transforms.ColorJitter(brightness = colour_jitter_level,
                                                                                      contrast = colour_jitter_level,
                                                                                      saturation = colour_jitter_level,
                                                                                      hue = colour_jitter_level),
                                                   torchvision.transforms.RandomRotation(degrees = degrees,
                                                                                         fill = self.value)])

    @staticmethod
    def get_params(img: Tensor,
                   scale: Tuple[float, float],
                   ratio: Tuple[float, float],
                   value: Optional[List[float]] = None):
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        # area = img_h * img_w
        #
        # log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            # erase_area = area * torch.empty(1).uniform_(scale[0],
            #                                             scale[1]).item()
            # aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0],
            #                                                  log_ratio[1])).item()

            h = torch.randint(low = 2,
                              high = 16,
                              size = (1,)).item()
            w = torch.randint(low = 10,
                              high = 25,
                              size = (1,)).item()
            if not (h < img_h and w < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w],
                                dtype = torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]
            i = torch.randint(0,
                              img_h - h + 1,
                              size = (1,)).item()
            j = torch.randint(0,
                              img_w - w + 1,
                              size = (1,)).item()
            return i, j, h, w, v

        # Return original image
        return 0, 0, img_h, img_w, img


class BYOLCutPasteAffine(torchvision.transforms.RandomErasing):

    def __init__(self,
                 apply_probability: float,
                 scale=(0.02, 0.15),
                 ratio=(0.3, 1),
                 colour_jitter_level: float = 0.1,
                 degrees = 30,
                 translate = (0.1, 0.1),
                 shear = (-15, 15)):
        super().__init__(p = apply_probability,
                         scale = scale,
                         ratio = ratio,
                         value = 255,
                         inplace = False)
        self.aug = torchvision.transforms.Compose([torchvision.transforms.ColorJitter(brightness = colour_jitter_level,
                                                                                      contrast = colour_jitter_level,
                                                                                      saturation = colour_jitter_level,
                                                                                      hue = colour_jitter_level),
                                                   torchvision.transforms.RandomAffine(degrees = degrees,
                                                                                       translate = translate,
                                                                                       shear = shear,
                                                                                       interpolation = InterpolationMode.NEAREST,
                                                                                       fill = self.value)])

    def forward(self,
                img):
        if torch.rand(1) < self.p:
            # print("CUT PASTE AFFINE")
            # cast self.value to script acceptable type
            if isinstance(self.value,
                          (int, float)):
                value = [float(self.value)]
            elif isinstance(self.value,
                            str):
                value = None
            elif isinstance(self.value,
                            (list, tuple)):
                value = [float(v) for v in self.value]
            else:
                value = self.value

            if value is not None and not (len(value) in (1, img.shape[-3])):
                raise ValueError(
                        "If value is a sequence, it should have either a single value or "
                        f"{img.shape[-3]} (number of input channels)"
                        )

            img_h, img_w = img.shape[-2], img.shape[-1]
            center_i, center_j, h, w, v, h_2, w_2 = self.get_params(img,
                                                                    scale = self.scale,
                                                                    ratio = self.ratio,
                                                                    value = value)
            if h == img_h or w == img_w:
                print("Warning, did not find a solution for cutpaste. Skipping")
                return img
            half_height = int(h / 2)
            half_width = int(w / 2)
            aug_img = self.aug(torchvision.transforms.functional.resize(img[...,
                                                                        center_i - half_height: center_i + half_height,
                                                                        center_j - half_width: center_j + half_width],
                                                                        size = [h_2,
                                                                                w_2],
                                                                        interpolation = InterpolationMode.BILINEAR))

            img = img.clone()
            i = center_i - math.ceil(h_2 / 2)
            j = center_j - math.ceil(w_2 / 2)
            img[..., i: i + h_2, j:j + w_2][aug_img != v] = aug_img[aug_img != v]
        return img

    @staticmethod
    def get_params(img: Tensor,
                   scale: Tuple[float, float],
                   ratio: Tuple[float, float],
                   value: Optional[List[float]] = None):
        """Get parameters for ``erase`` for a random erasing.

        Args:
            img (Tensor): Tensor image to be erased.
            scale (sequence): range of proportion of erased area against input image.
            ratio (sequence): range of aspect ratio of erased area.
            value (list, optional): erasing value. If None, it is interpreted as "random"
                (erasing each pixel with random values). If ``len(value)`` is 1, it is interpreted as a number,
                i.e. ``value[0]``.

        Returns:
            tuple: params (i, j, h, w, v) to be passed to ``erase`` for random erasing.
        """
        img_c, img_h, img_w = img.shape[-3], img.shape[-2], img.shape[-1]
        area = img_h * img_w

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            erase_area = area * torch.empty(1).uniform_(scale[0],
                                                        scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0],
                                                             log_ratio[1])).item()

            h = int(round(math.sqrt(erase_area * aspect_ratio)))
            w = int(round(math.sqrt(erase_area / aspect_ratio)))

            erase_area_2 = area * torch.empty(1).uniform_(scale[0],
                                                          scale[1]).item()
            aspect_ratio_2 = torch.exp(torch.empty(1).uniform_(log_ratio[0],
                                                               log_ratio[1])).item()

            h_2 = int(round(math.sqrt(erase_area_2 * aspect_ratio_2)))
            w_2 = int(round(math.sqrt(erase_area_2 / aspect_ratio_2)))

            if not (h < img_h and w < img_w and h_2 < img_h and w_2 < img_w):
                continue

            if value is None:
                v = torch.empty([img_c, h, w],
                                dtype = torch.float32).normal_()
            else:
                v = torch.tensor(value)[:, None, None]

            half_max_width = math.ceil(max(w_2,
                                     w) / 2)
            half_max_height = math.ceil(max(h_2,
                                      h) / 2)
            center_i = torch.randint(half_max_height,
                                     img_h - half_max_height + 1,
                                     size = (1,)).item()
            center_j = torch.randint(half_max_width + 1,
                                     img_w - half_max_width + 1,
                                     size = (1,)).item()
            return center_i, center_j, h, w, v, h_2, w_2

        # Return original image
        return 0, 0, img_h, img_w, img, None, None
