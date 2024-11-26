import os
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, RandomSampler
from torchvision.models._utils import IntermediateLayerGetter

from navbev.config import Configs as cfg
from navbev.perception.backend.dataloader import vocabulary
from navbev.perception.backend.losses import Loss
from navbev.perception.backend.models.model import JointModel
from navbev.perception.glc import train, transforms

torch.manual_seed(12)
torch.cuda.manual_seed(12)
random.seed(12)
np.random.seed(12)

torch.backends.cudnn.benchmark = True

torch.cuda.empty_cache() 

_conf = cfg.glc()


class Args:
    def __init__(self, mask_thresh, mask_dim, loss):
        self.mask_thresh = mask_thresh
        self.topk = 1
        self.mask_dim = mask_dim
        self.loss = loss


class GLCModel:
    def __init__(self):
        self.device = cfg.globals().device
        self.mask_dim = _conf.mask.size
        self.mask_thresh = _conf.mask.thresh
        self.loss = _conf.train.loss

        self.args = Args(self.mask_thresh, self.mask_dim, self.loss)
        self.loss_func = Loss(self.args)

        self._load_image_encoder()
        self._load_joint_model()

        param_dicts = [
            {"params": [p for p in self.joint_model.parameters() if p.requires_grad]},
        ]

        self.optimizer = torch.optim.Adam(
            param_dicts, lr=_conf.train.lr, weight_decay=_conf.train.wd
        )

    def _load_image_encoder(self):
        model = torch.hub.load(
            "pytorch/vision:v0.10.0", f"{_conf.image_enc.model}", pretrained=True
        )

        model.load_state_dict(
            torch.load(
                os.path.expanduser(
                    f"~/.cache/torch/hub/checkpoints/{_conf.paths.deeplab_chkpt}.pth"
                )
            )
        )
        self.return_layers = {
            "layer2": "layer2",
            "layer3": "layer3",
            "layer4": "layer4",
        }
        self.image_encoder = IntermediateLayerGetter(model.backbone, self.return_layers)

        if self.image_encoder:
            self.image_encoder.to(self.device)
            print(f"\nLoaded {_conf.image_enc.model.upper()}")

        for param in self.image_encoder.parameters():
            param.requires_grad_(False)
            self.image_encoder.eval()

    def _load_joint_model(self):
        self.joint_model = JointModel(
            in_channels=_conf.model.inp_dim,
            out_channels=_conf.model.out_dim,
            stride=_conf.model.stride,
            num_layers=_conf.model.n_layers,
            num_encoder_layers=_conf.model.enc_layers,
            dropout=_conf.model.dropout,
            skip_conn=_conf.model.skip,
            mask_dim=self.mask_dim,
            vocab_size=_conf.text_enc.vocab_size,
        )
        
        glc = 'glc'
        self.joint_model.to(self.device)
        self.joint_model.load_state_dict(
            # torch.load(
            #     f"{cfg.ROOT}/{cfg.globals().paths.checkpoint_path}/{_conf.paths.glove_dir}/{_conf.paths.goalpred_chkpt}.pth"
            # )
            torch.load(
                f"{cfg.ROOT}/{cfg.globals().paths.checkpoint_path}/{glc}/{_conf.paths.goalpred_chkpt}.pth"
            )["state_dict"],
            strict=False,
        )
        
        # self.joint_model.load_state_dict(
        #     torch.load(
        #         f"{cfg.ROOT}/{cfg.globals().paths.checkpoint_path}/e2e/e2e_glc_{cfg.e2e().train.epochs}.pth",
        #     )
        # )
        
        if self.joint_model:
            print(f"Loaded {_conf.paths.goalpred_chkpt.upper()}")

        for param in self.joint_model.parameters():
            param.requires_grad = True

        for param in self.joint_model.goal_prediction.parameters():
            param.requires_grad = True

    def train_model(self):
        save_path = "./saved_model"
        for epochId in range(_conf.train.n_epochs):
            model_filename = os.path.join(save_path, str(epochId) + ".pth")
            train.train(
                self.train_loader,
                self.joint_model,
                self.image_encoder,
                self.optimizer,
                epochId,
            )
            if epochId % 2 == 0:
                torch.save(
                    {
                        "epoch": epochId,
                        "state_dict": self.joint_model.state_dict(),
                        "optimizer": self.optimizer.state_dict(),
                    },
                    model_filename,
                )


class GLCInference:
    def __init__(self):
        self.glc = GLCModel()
        self.feat_dim = cfg.glc().infer.feat_dim

        self._vocab = vocabulary.Vocabulary(
            voc_file=f"{cfg.PKG}/perception/backend/dataloader/vocabulary.txt",
            glove_path=f"{cfg.ROOT}/dataset/glc_data/glove",
            max_len=_conf.text_enc.seq_length,
        )

        self._tf_image, _ = transforms.get_transforms(cfg.glc().infer.resize)

        self.glc.joint_model.eval()
        self.glc.image_encoder.eval()

    def _normalize_goal(self, goal):

        goal[0] = goal[0] * cfg.glc().normalize_params.goal.sigma_x + cfg.glc().normalize_params.goal.mu_x
        goal[1] = goal[1] * cfg.glc().normalize_params.goal.sigma_y + cfg.glc().normalize_params.goal.mu_y
        
        goal[0] *= self._W / cfg.glc().infer.resize
        goal[1] *= self._H / cfg.glc().infer.resize
        return goal

    def predict_goal(self, image: Image.Image, text: str):
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        self._H, self._W = np.shape(np.array(image))[:2]

        with torch.no_grad():
            img = self._tf_image(image).cuda()

            phrase, phrase_mask = self._vocab.tokenize(text)
            phrase = phrase.unsqueeze(0).cuda()
            phrase_mask = phrase_mask.unsqueeze(0).cuda()
            img_mask = torch.ones(
                1, self.feat_dim * self.feat_dim, dtype=torch.int64
            ).cuda()

            img = self.glc.image_encoder(img.unsqueeze(0))

            _, goal_2d = self.glc.joint_model(img, phrase, img_mask, phrase_mask)

        return self._normalize_goal(goal=goal_2d.squeeze(0)).detach().cpu().numpy()
