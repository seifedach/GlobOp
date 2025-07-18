import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def SuGD(model, criterion, batch, low_bound, up_bound, device, sugd_iters = 2, w = 0.2, output = "up", history_dir = False):
    x_sample, y_sample = batch
    x_sample, y_sample = x_sample.to(device), y_sample.to(device)

    # Save weights once
    weights_backup = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    
    lr1, lr2 = low_bound, up_bound
    lr1s, lr2s = [], []
    w = w

    for _ in range(sugd_iters):
        # === LR1 Step ===
        model.load_state_dict(weights_backup)
        for param in model.parameters():
            param.requires_grad = True
        optimizer1 = optim.Adam(model.parameters(), lr=lr1)
        model.train()
        optimizer1.zero_grad()
        outputs1 = model(x_sample)
        loss1 = criterion(outputs1, y_sample)
        loss1.backward()
        optimizer1.step()

        # Evaluate loss after LR1 step
        model.eval()
        with torch.no_grad():
            val_loss1 = criterion(model(x_sample), y_sample).item()

        # === LR2 Step ===
        model.load_state_dict(weights_backup)
        optimizer2 = optim.Adam(model.parameters(), lr=lr2)
        model.train()
        optimizer2.zero_grad()
        outputs2 = model(x_sample)
        loss2 = criterion(outputs2, y_sample)
        loss2.backward()
        optimizer2.step()

        # Evaluate loss after LR2 step
        model.eval()
        with torch.no_grad():
            val_loss2 = criterion(model(x_sample), y_sample).item()

        # Update bounds
        delta = (val_loss2 - val_loss1)
        if delta < 0:
            lr1 = lr1 - w * (lr2 - lr1) * delta
        else:
            lr2 = lr2 - w * (lr2 - lr1) * delta

        # Clean up memory
        del optimizer1, optimizer2, outputs1, outputs2, loss1, loss2
    print(lr1, lr2)
    model.load_state_dict(weights_backup)
    if output == "up":
        return float(lr2)
    elif output == "down":
        return float(lr1)
    elif output == "avg":
        return float((lr1 + lr2) / 2)
