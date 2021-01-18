from tqdm import tqdm


def train_fn(data_loader, model, optimizer, device, accumulatiion_steps):
    model.train()

    for bi, d in tqdm(enumerate(data_loader), total=len(data_loader)):
        pass


def eval_fn():
    pass
