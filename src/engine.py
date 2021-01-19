from tqdm import tqdm
import torch.nn as nn
import torch


def loss_fn(outputs, targets):
    # TODO attention target format size
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, device, accumulatiion_steps):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        pass


def eval_fn():
    pass
