import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable

mean=[0.389, 0.397, 0.393]
std=[0.221, 0.215, 0.217]

class ResizeAnnotation:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        im_h, im_w = img.shape[-2:]
        scale_h, scale_w = self.size / im_h, self.size / im_w
        resized_h = int(np.round(im_h * scale_h))
        resized_w = int(np.round(im_w * scale_w))
        out = (
            F.interpolate(
                Variable(img).unsqueeze(0).unsqueeze(0),
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=True,
            )
            .squeeze()
            .data
        )
        return out

def get_transforms(mask_dim):
    normalize = transforms.Normalize(
        mean, std
    )
    to_tensor = transforms.ToTensor()
    resize = transforms.Resize((mask_dim, mask_dim))
    random_grayscale = transforms.RandomGrayscale(p=0.3)

    image_transform = transforms.Compose([resize, to_tensor, normalize])
    mask_transform = transforms.Compose([ResizeAnnotation(mask_dim)])

    return image_transform, mask_transform


def unnormalize_goal(cfg, arr):
    mu_x = 199.480
    mu_y = 272.540
    sigma_x = 50.808
    sigma_y = 15.042
    arr[:, 0] = arr[:, 0] * sigma_x + mu_x
    arr[:, 1] = arr[:, 1] * sigma_y + mu_y
    arr[:, 0] = arr[:, 0] * (cfg.glc().image.W / 448.)
    arr[:, 1] = arr[:, 1] * (cfg.glc().image.H / 448.)
    return arr


def normalize_image(image_tensor):
    min_val = torch.min(image_tensor)
    shifted_image = image_tensor - min_val
    max_val = torch.max(shifted_image)
    return shifted_image / max_val


def unnormalize_image(image_tensor):
    mean_tensor = torch.tensor(mean).reshape(1, 1, len(mean))
    std_tensor = torch.tensor(std).reshape(1, 1, len(std))
    return image_tensor * std_tensor + mean_tensor
