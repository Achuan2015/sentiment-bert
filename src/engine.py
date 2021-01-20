# from tqdm import tqdm
from tqdm.auto import tqdm
import torch.nn as nn
import torch


def loss_fn(outputs, targets):
    # TODO attention target format size
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets = d['targets']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        # clear out gradient
        optimizer.zero_grad()
        outputs = model(ids=ids, token_type_ids=token_type_ids, mask=mask)
        loss = loss_fn(outputs, targets)
        # backpropagation process: calcualte gradient
        loss.backward()

        optimizer.step()
        scheduler.step()

def eval_fn(data_loader, model, device):
    model.eval()

    fin_outputs = []
    fin_targets = []
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d['ids']
        token_type_ids = d['token_type_ids']
        mask = d['mask']
        targets = d['targets']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(ids=ids, token_type_ids=token_type_ids, mask=mask)
        fin_targets.extend(targets.cpu().detach().numpy().tolist())
        fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets
