import torch

def train(
    train_loader,
    joint_model,
    image_encoder,
    optimizer,
    loss_func,
    epochId,
    args,
):
    joint_model.train()
    image_encoder.eval()
    optimizer.zero_grad()

    feature_dim = 14
    data_len = train_loader.dataset.__len__()

    mse_loss = torch.nn.MSELoss()

    if epochId == 0:
        print(f"Train data length: {data_len}")

    for step, batch in enumerate(train_loader):
        iterId = step + (epochId * data_len) - 1
        print("Iter: ", iterId)

        with torch.no_grad():
            img = batch["image"].cuda()
            phrase = batch["phrase"].cuda()
            phrase_mask = batch["phrase_mask"].cuda()

            gt_mask = batch["seg_mask"].cuda()
            gt_mask = gt_mask.squeeze(dim=1)

            gp_gt = batch["goal_position"].cuda().float()
            nan_mask = ~torch.isnan(gp_gt)

            batch_size = img.shape[0]
            img_mask = torch.ones(
                batch_size, feature_dim * feature_dim, dtype=torch.int64
            ).cuda()

            img = image_encoder(img)

        _, goal_2d = joint_model(img, phrase, img_mask, phrase_mask)

        loss = mse_loss(goal_2d, gp_gt)
        print("Loss: ", loss.item())
        loss.backward()

        optimizer.step()
        joint_model.zero_grad()
