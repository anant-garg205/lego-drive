import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, RandomSampler

from navbev.config import Configs as cfg
from navbev.perception.backend.dataloader import talk2car, vocabulary
from navbev.perception.glc import transforms

_conf = cfg.glc()

vocab = vocabulary.Vocabulary(
    voc_file=f"{cfg.PKG}/perception/backend/dataloader/vocabulary.txt",
    glove_path=f"{cfg.ROOT}/{cfg.globals().paths.checkpoint_path}/{cfg.glc().paths.glove_dir}",
    max_len=cfg.glc().text_enc.seq_length,
)


def prepare_dataloader():
    image_transf, mask_transf = transforms.get_transforms(mask_dim=448)

    train_dataset = talk2car.Talk2Car(
        root=None,
        split="train",
        transform=image_transf,
        mask_transform=mask_transf,
        glove_path=os.path.expanduser(f"~/{_conf.paths.glove_dir}"),
        max_len=_conf.text_enc.seq_length,
    )

    train_sampler = RandomSampler(train_dataset)

    return DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=_conf.train.batch,
        num_workers=_conf.train.n_workers,
        pin_memory=True,
        drop_last=True,
    )


class GLCDataset(Dataset):
    def __init__(self, image_dir, mask_dir, json_file):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_tf, self.mask_tf = transforms.get_transforms(mask_dim=448)

        with open(json_file, "r") as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data.keys())

    def __getitem__(self, idx):
        idx = str(idx).zfill(4)
        image_path = f"{self.image_dir}/{idx}.png"
        mask_path = f"{self.mask_dir}/{idx}.png"
        text = self.data.get(idx)

        phrase, phrase_mask = vocab.tokenize(text)

        image = Image.open(image_path)
        original_image = torch.from_numpy(np.array(image))
        image = self.image_tf(image)

        mask_img = Image.open(mask_path).convert("L")
        mask_img = torch.from_numpy(np.array(mask_img))

        mask_img = mask_img.float()
        mask_img = self.mask_tf(mask_img)
        mask = torch.zeros_like(mask_img)
        mask[mask_img > 0] = 1

        return original_image, image, mask, phrase, phrase_mask


if __name__ == "__main__":
    dataset = GLCDataset(
        image_dir=f"{cfg.ROOT}/dataset/planner_data/images",
        mask_dir=f"{cfg.ROOT}/dataset/planner_data/masks_rgb",
        json_file=f"{cfg.ROOT}/dataset/planner_data/commands.json",
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=True,
        num_workers=_conf.train.n_workers,
        drop_last=True,
    )

    for k, data in enumerate(dataloader, start=1):
        image, mask, phrase, phrase_mask = data
        print(image.shape)  # (B, 3, 448, 488)
        print(mask.shape)  # (B, 448, 488)
        print(phrase.shape)  # (B, 40, 300)
        print(phrase_mask.shape)  # (B, 40)
        break
