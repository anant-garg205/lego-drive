import os
import psutil
import gc
from time import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import print_, grad_check
from utils.metrics import *


def train(
    train_loader,
    joint_model,
    image_encoder,
    optimizer,
    loss_func,
    # experiment,
    epochId,
    args,
):
    pid = os.getpid()
    py = psutil.Process(pid)

    joint_model.train()
    image_encoder.eval()
    optimizer.zero_grad()

    total_loss = 0
    total_accuracy = 0
    total_inter, total_union = 0, 0

    feature_dim = 14

    epoch_start = time()

    n_iter = 0
    data_len = train_loader.dataset.__len__()

    mse_loss = nn.MSELoss()

    if epochId == 0:
        print(f"Train data length: {data_len}")

    for step, batch in enumerate(train_loader):
        iterId = step + (epochId * data_len) - 1
        print("Iter: ", iterId)

        with torch.no_grad():
            img = batch["image"].cuda(non_blocking=True)
            phrase = batch["phrase"].cuda(non_blocking=True)
            phrase_mask = batch["phrase_mask"].cuda(non_blocking=True)

            gt_mask = batch["seg_mask"].cuda(non_blocking=True)
            gt_mask = gt_mask.squeeze(dim=1)

            gp_gt = batch["goal_position"].cuda(non_blocking=True).float()
            # gp_gt = F.normalize(gp_gt)

            nan_mask = ~torch.isnan(gp_gt)

            # print("Goal GT: ",gp_gt)

            batch_size = img.shape[0]
            img_mask = torch.ones(
                batch_size, feature_dim * feature_dim, dtype=torch.int64
            ).cuda(non_blocking=True)

        start_time = time()

        with torch.no_grad():
            img = image_encoder(img)
        mask, goal_2d = joint_model(img, phrase, img_mask, phrase_mask)

        goal_2d = goal_2d[nan_mask]
        gp_gt = gp_gt[nan_mask]

        # print("Goal prediction: ", goal_2d.type(), "goal gt: ", gp_gt.type())
        loss = mse_loss(goal_2d, gp_gt)
        print("Loss: ", loss.item())
        loss.backward()

        # loss = loss_func(mask, gt_mask)

        # loss.backward()

        # if iterId % 500 == 0 and args.grad_check:
        #     grad_check(joint_model.named_parameters(), experiment)

        optimizer.step()
        joint_model.zero_grad()

        end_time = time()
        elapsed_time = end_time - start_time

        inter, union = compute_batch_IOU(mask, gt_mask, mask_thresh=args.mask_thresh)

        total_inter += inter.sum().item()
        total_union += union.sum().item()

        total_accuracy += pointing_game(mask, gt_mask)

        n_iter += batch_size

        total_loss += float(loss.item())

        if iterId % 50 == 0 and step != 0:
            gc.collect()
            memoryUse = py.memory_info()[0] / 2.0**20

            timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

            curr_loss = total_loss / (step + 1)
            curr_IOU = total_inter / total_union
            curr_acc = total_accuracy / n_iter

            lr = optimizer.param_groups[0]["lr"]

            print_(
                f"{timestamp} Epoch:[{epochId:2d}/{args.epochs:2d}] iter {iterId:6d} loss {curr_loss:.4f} IOU {curr_IOU:.4f} accuracy {curr_acc:.4f} memory_use {memoryUse:.3f}MB lr {lr:.7f} elapsed {elapsed_time:.2f}"
            )

    epoch_end = time()
    epoch_time = epoch_end - epoch_start

    timestamp = datetime.now().strftime("%Y|%m|%d-%H:%M")

    train_loss = total_loss / data_len
    train_IOU = total_inter / total_union
    train_accuracy = total_accuracy / data_len

    # experiment.log(
    #     {"train_loss": train_loss, "train_IOU": train_IOU, "train_Accuracy": train_accuracy}
    # )
    print(
        "train loss: {0}, train_IOU: {1}, Train_Accuracy: {2}".format(
            train_loss, train_IOU, train_accuracy
        )
    )

    print_(
        f"{timestamp} FINISHED Epoch:{epochId:2d} loss {train_loss:.4f} IOU {train_IOU:.4f} Accuracy {train_accuracy:.4f} elapsed {epoch_time:.2f}"
    )
