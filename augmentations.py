from torch.distributions import Uniform
from torchvision.transforms.functional import adjust_hue, adjust_brightness, adjust_contrast, adjust_saturation

class BYOLColorJitter():

    def __init__(self,
                 max_brightness=0.5,
                 max_contrast=0.5,
                 max_saturation=0.2,
                 max_hue=0.1):
        self.colour_jitter_components = [(adjust_hue, max_hue),
                                         (adjust_brightness, max_brightness),
                                         (adjust_contrast, max_contrast),
                                         (adjust_saturation, max_saturation)]





    def __call__(self,
                 image):
        random.shuffle(self.colour_jitter_components)
        for component, adjustment_max_value in self.colour_jitter_components:
            if component == adjust_hue:
                no_change_val = 0
            else:
                no_change_val = 1
            adjustment_val = Uniform(no_change_val-adjustment_max_value, no_change_val+adjustment_max_value).sample()
            image = component(image, adjustment_val)

        return image

