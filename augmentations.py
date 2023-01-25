import random

import torch
import torchvision.transforms
from  torch.distributions.uniform import Uniform


class BYOLAug:

    def __init__(self,
                 image_aug_prob):
        self.image_aug_probs = image_aug_prob
        self.aug = None

    def __call__(self,
                 image_batch):
        batch_size = image_batch.shape[0]
        probabilities = torch.rand(size = (batch_size,))
        for index in range(batch_size):
            if probabilities[index] < self.image_aug_probs:
                image_batch[index] = self.aug(image_batch[index])

        return image_batch

class BYOLResizedCrop(BYOLAug):

    def __init__(self,
                 image_aug_prob = 1.0,
                 output_size=(32, 32),
                 scale=(0.08, 1.0),
                 ratio=(3 / 4, 4 / 3)):
        super().__init__(image_aug_prob)
        self.aug = torchvision.transforms.RandomResizedCrop(size = output_size,
                                                            scale = scale,
                                                            ratio = ratio)


class BYOLHorizontalFlip(BYOLAug):

    def __init__(self,
                 image_aug_prob=0.5):
        super().__init__(image_aug_prob)
        self.aug = torchvision.transforms.functional.hflip

class BYOLColorJitter(BYOLAug):

    def __init__(self,
                 image_aug_prob=0.8,
                 max_brightness=0.5,
                 max_contrast=0.5,
                 max_saturation=0.2,
                 max_hue=0.1):
        super().__init__(image_aug_prob)

        self.colour_jitter_components = [(torchvision.transforms.functional.adjust_hue, max_hue),
                                         (torchvision.transforms.functional.adjust_brightness, max_brightness),
                                         (torchvision.transforms.functional.adjust_contrast, max_contrast),
                                         (torchvision.transforms.functional.adjust_saturation, max_saturation)]





    def __call__(self,
                 image_batch):
        batch_size = image_batch.shape[0]
        probabilities = torch.rand(size = (batch_size,))
        for index in range(batch_size):
            if probabilities[index] < self.image_aug_probs:
                random.shuffle(self.colour_jitter_components)
                for component, adjustment_max_value in self.colour_jitter_components:

                    if component == torchvision.transforms.functional.adjust_hue:
                        no_change_val = 0
                    else:
                        no_change_val = 1
                    adjustment_val = Uniform(no_change_val-adjustment_max_value, no_change_val+adjustment_max_value).sample()
                    image_batch[index] = component(image_batch[index], adjustment_val)


        return image_batch


class BYOLGaussianBlur(BYOLAug):

    def __init__(self,
                 image_aug_prob):
        super().__init__(image_aug_prob)
        self.aug = torchvision.transforms.GaussianBlur(kernel_size = 23,
                                                       sigma=(0.1, 2.0))

class BYOLSolarisation(BYOLAug):

    def __init__(self,
                 image_aug_prob):
        super().__init__(image_aug_prob)
        #TODO : Comments / alternate way around this approach
        self.aug = torchvision.transforms.RandomSolarize(threshold = 0.5, p = 1.0)



#
# class BYOLResizedCrop(BYOLAug):
#
#     def __init__(self,
#                  image_aug_probs=(1.0, 1.0),
#                  output_size=(224, 224),
#                  scale=(0.08, 1.0),
#                  ratio=(3 / 4, 4 / 3)):
#         super().__init__(image_aug_probs)
#         self.aug = torchvision.transforms.RandomResizedCrop(size = output_size,
#                                                             scale = scale,
#                                                             ratio = ratio)

#
# class BYOLHorizontalFlip(BYOLAug):
#
#     def __init__(self,
#                  image_aug_probs=(0.5, 0.5)):
#         super().__init__(image_aug_probs)
#         self.aug = torchvision.transforms.functional.hflip
#
#
# class BYOLColorJitter(BYOLAug):
#
#     def __init__(self,
#                  image_aug_probs=(0.8, 0.8),
#                  max_brightness=(0.4, 0.4),
#                  max_contrast=(0.4, 0.4),
#                  max_saturation=(0.2, 0.2),
#                  max_hue=(0.1, 0.1)):
#         super().__init__(image_aug_probs)
#         self.image_aug_probs = image_aug_probs
#         self.image_1_max_brightness, self.image_2_max_brightness = max_brightness
#         self.image_1_max_contrast, self.image_2_max_contrast = max_contrast
#         self.image_1_max_saturation, self.image_2_max_saturation = max_saturation
#         self.image_1_max_hue, self.image_2_hue = max_hue
#
#     def __call__(self,
#                  image_1,
#                  image_2):
#         if image_1.shape != image_2.shape:
#             raise Exception("Both image batches must have the same shape ")
#         images = (image_1, image_2)
#         batch_size = image_1.shape[0]
#         probabilities = torch.rand(size = (2, batch_size))
#         for index in range(batch_size):
#             for prob, image, min_prob in zip(probabilities[index], images, self.image_aug_probs):
#                 if prob < min_prob:
#
#
#
#
#


