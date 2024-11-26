import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models._utils import IntermediateLayerGetter
from models.modeling.deeplab import *
from PIL import Image
from skimage.transform import resize
from models.model import JointModel
from dataloader.vocabulary import Vocabulary
import cv2

import matplotlib.pyplot as plt


class Args:
    lr = 3e-4
    batch_size = 64
    num_workers = 4
    image_encoder = "deeplabv3_plus"
    num_layers = 1
    num_encoder_layers = 1
    dropout = 0.25
    skip_conn = False
    model_path = "/scratch/anant.garg/55.pth"
    dataroot = "/scratch/anant.garg/Talk2Car-RefSeg"
    glove_path = "/scratch/anant.garg/glove"
    vocabulary_path = (
        "/home/anant.garg/Documents/iros_24/code/GLC/dataloader/vocabulary.txt"
    )
    dataset = "talk2car"
    task = "talk2car"
    split = "val"
    max_len = 40
    image_dim = 448
    mask_dim = 448
    mask_thresh = 0.3
    area_thresh = 0.4
    topk = 10
    metric = "pointing_game"


class GLC:
    def __init__(self):
        self.args = Args()
        self.device = torch.cuda("cuda:0" if torch.cuda.is_available() else "cpu")
        self.vocabulary = Vocabulary(
            self.args.vocabulary_path, self.args.glove_path, self.args.max_len
        )

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        to_tensor = transforms.ToTensor()
        resize = transforms.Resize((self.args.image_dim, self.args.image_dim))

        self.transform = transforms.Compose([resize, to_tensor, normalize])

    def configureModel(self):
        return_layers = {"layer2": "layer2", "layer3": "layer3", "layer4": "layer4"}

        model = torch.hub.load(
            "pytorch/vision:v0.10.0", "deeplabv3_resnet50", pretrained=True
        )
        model.load_state_dict(
            torch.load(
                "/home/anant.garg/.cache/torch/hub/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth"
            )
        )

        self.image_encoder = IntermediateLayerGetter(model.backbone, return_layers)

        for param in self.image_encoder.parameters():
            param.requires_grad_(False)

        in_channels = 2048
        out_channels = 512
        stride = 2

        self.joint_model = JointModel(
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            num_layers=self.args.num_layers,
            num_encoder_layers=self.args.num_encoder_layers,
            dropout=self.args.dropout,
            skip_conn=self.args.skip_conn,
            mask_dim=self.args.mask_dim,
        )

        state_dict = torch.load(self.args.model_path)
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        self.joint_model.load_state_dict(state_dict)

        self.joint_model.to(self.device)
        self.image_encoder.to(self.device)

        self.image_encoder.eval()
        self.joint_model.eval()

    def runstep(self, input_image, command):
        image_dims = [input_image.shape[0], input_image.shape[1]]
        image = self.transform(input_image)
        phrase, phrase_mask = self.vocabulary.tokenize(command)
        image = image.cuda(non_blocking=True).unsqueeze(0)
        image_mask = torch.ones(1, 14 * 14, dtype=torch.int64).cuda(non_blocking=True)
        phrase = phrase.unsqueeze(0).cuda(non_blocking=True)
        phrase_mask = phrase_mask.unsqueeze(0).cuda(non_blocking=True)

        with torch.no_grad():
            image = self.image_encoder(image)

        output_mask, goal_2d = self.joint_model(image, phrase, image_mask, phrase_mask)

        output_mask = output_mask.detach().cpu().squeeze()
        goal_2d = goal_2d.detach().cpu().numpy()[0]

        goal_2d[0] = goal_2d[0] * 104.50883928 + 212.26290574
        goal_2d[1] = goal_2d[1] * 57.59293929 + 305.78739969

        goal_2d[0] *= image_dims[0] / 448
        goal_2d[1] *= image_dims[1] / 448

        return goal_2d
