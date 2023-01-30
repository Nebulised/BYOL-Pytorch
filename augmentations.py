import random

import torchvision.transforms
from torchvision.transforms import ToTensor, Normalize
from torch.distributions import Uniform
from torchvision.transforms.functional import adjust_hue, adjust_brightness, adjust_contrast, adjust_saturation, InterpolationMode
import torch


class BYOLAugmenter():

    def __init__(self,
                 view_1_params,
                 view_2_params,
                 fine_tune_params,
                 test_params,
                 model_input_height,
                 model_input_width):
        self.model_input_height = model_input_height
        self.model_input_width = model_input_width
        for params in (view_1_params, view_2_params, fine_tune_params):
            params["resize_crop"]["output_height"] = model_input_height
            params["resize_crop"]["output_width"] = model_input_width
        self.view_1 = self.create_view(**view_1_params)
        self.view_2 = self.create_view(**view_2_params)
        self.fine_tune_augs = self.get_fine_tune_augmentations(**fine_tune_params)
        self.test_augs = self.get_test_augmentations(**test_params)



    def self_supervised_pre_train_transform(self, image):
        return self.view_1(image), self.view_2(image)




    def get_fine_tune_augmentations(self,
                                    resize_crop,
                                    random_flip,
                                    normalise):
        return torchvision.transforms.Compose([BYOLRandomResize(**resize_crop),
                                               BYOLHorizontalFlip(**random_flip),
                                               ToTensor(),
                                               Normalize(**normalise)])


    def get_test_augmentations(self, normalise):
        return torchvision.transforms.Compose([torchvision.transforms.Resize(size=(self.model_input_height,
                                                                                   self.model_input_width),
                                                                             interpolation = InterpolationMode.BICUBIC),
                                               ToTensor(),
                                               Normalize(**normalise)])

    @staticmethod
    def create_view(resize_crop,
                    random_flip,
                    colour_jitter,
                    gaussian_blur,
                    solarize,
                    normalise,
                    colour_drop):
        view_augs = []
        view_augs.append(BYOLRandomResize(**resize_crop))
        view_augs.append(BYOLHorizontalFlip(**random_flip))
        view_augs.append(BYOLRandomColourJitter(**colour_jitter))
        view_augs.append(BYOLColourDrop(**colour_drop))
        view_augs.append(BYOLGaussianBlur(**gaussian_blur))
        view_augs.append(BYOLSolarize(**solarize))
        view_augs.append(ToTensor())
        view_augs.append(Normalize(**normalise))

        return torchvision.transforms.Compose(view_augs)


class BYOLRandomApplyAug():
    def __init__(self,
                 apply_probability):
        self.apply_probability = apply_probability
        self.aug = None

    def __call__(self,
                 image):
        if random.random() < self.apply_probability:
            image = self.aug(image)

        return image


class BYOLRandomResize(BYOLRandomApplyAug):

    def __init__(self,
                 apply_probability,
                 output_height,
                 output_width):
        super().__init__(apply_probability)
        self.aug = torchvision.transforms.RandomResizedCrop(size = (output_height, output_width),
                                                            interpolation = InterpolationMode.BICUBIC)


class BYOLHorizontalFlip(BYOLRandomApplyAug):

    def __init__(self,
                 apply_probability):
        super().__init__(apply_probability)
        self.aug = torchvision.transforms.functional.hflip


class BYOLRandomColourJitter(BYOLRandomApplyAug):

    def __init__(self,
                 apply_probability,
                 brightness_delta,
                 contrast_delta,
                 saturation_delta,
                 hue_delta):
        super().__init__(apply_probability)
        self.aug = BYOLColourJitter(brightness_delta = brightness_delta,
                                    contrast_delta = contrast_delta,
                                    saturation_delta = saturation_delta,
                                    hue_delta = hue_delta)


class BYOLGaussianBlur(BYOLRandomApplyAug):

    def __init__(self,
                 apply_probability,
                 kernel_size,
                 sigma):
        super().__init__(apply_probability)
        self.aug = torchvision.transforms.GaussianBlur(kernel_size = kernel_size,
                                                       sigma = sigma)


class BYOLSolarize(BYOLRandomApplyAug):

    def __init__(self,
                 apply_probability,
                 threshold):
        super().__init__(apply_probability)
        self.aug = torchvision.transforms.functional.solarize
        self.threshold = threshold

    def __call__(self,
                 image):
        if random.random() < self.apply_probability:
            image = self.aug(image,
                             self.threshold)
        return image



class BYOLColourJitter():

    def __init__(self,
                 brightness_delta=0.5,
                 contrast_delta=0.5,
                 saturation_delta=0.2,
                 hue_delta=0.1):
        self.colour_jitter_components = [(adjust_hue, hue_delta),
                                         (adjust_brightness, brightness_delta),
                                         (adjust_contrast, contrast_delta),
                                         (adjust_saturation, saturation_delta)]

    def __call__(self,
                 image):
        random.shuffle(self.colour_jitter_components)
        for component, adjustment_max_value in self.colour_jitter_components:
            if component == adjust_hue:
                no_change_val = 0
            else:
                no_change_val = 1
            adjustment_val = Uniform(no_change_val - adjustment_max_value,
                                     no_change_val + adjustment_max_value).sample()
            image = component(image,
                              adjustment_val)

        return image


class BYOLColourDrop(BYOLRandomApplyAug):

    def __init__(self,
                 apply_probability):
        super().__init__(apply_probability)
        self.rgb_converter = torch.Tensor([0.2989, 0.5870, 0.1140]).reshape(1,
                                                                            3,
                                                                            1)

    def __call__(self,
                 image):
        if random.random() < self.apply_probability:

            image = image.convert("L").convert("RGB")

        return image
