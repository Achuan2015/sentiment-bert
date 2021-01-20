from tqdm import tqdm
import torch.nn as nn
import torch


def loss_fn(outputs, targets):
    # TODO attention target format size
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))

def train_fn(data_loader, model, optimizer, device, accumulatiion_steps):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d['ids']
        token_type_id = d['ids']
        mask = d['mask']
        targets = d['targets']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = toekn_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)
        
        # clear out gradient
        optimzier.zero()
        outputs = model(ids=ids, token_type_ids=token_type_id, mask=mask)
        loss = loss_fn(outputs, targets)
        # backpropagation process: calcualte gradient
        loss.backward()

        optimizer.step()
        scheduler.step()

def eval_fn(data_loader, model):
    model.eval()

    fin_outputs = []
    fin_outputs = []
    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = d['ids']
        token_type_id = d['ids']
        mask = d['mask']
        targets = d['targets']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = toekn_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(ids=ids, token_type_ids=token_type_id, mask=mask)
        fin_outputs.extend(outputs.cpu().detach().numpy().tolist())
        fin_targets.extend(torch.sigmoid(targets).cpu().detach().numpy().tolist())
        return fin_outputs, fin_targets
