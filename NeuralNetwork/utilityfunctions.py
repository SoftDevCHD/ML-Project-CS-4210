import torch
import math


def encode_label(label):
    if label == 'Win':
        return 0
    elif label == 'Draw':
        return 1
    else:
        return 2


def decode_label(label):
    if label == 0:
        return 'Win'
    elif label == 1:
        return 'Draw'
    else:
        return 'Lose'


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def predict_winning_team(data, mod, device):
    x = to_device(data, device)
    predictions = mod(x)
    return predictions[0].item()


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, dev):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, dev) for x in data]
    return data.to(dev, non_blocking=True)


def evaluate(modl, val_loader):
    """Evaluate the model's performance on the validation set"""
    outputs = [modl.validation_step(batch) for batch in val_loader]
    return modl.validation_epoch_end(outputs)


def fit(num_epochs, lr, modl, train_load, val_load, opt_func=torch.optim.SGD):
    """Train the model using gradient descent"""
    history = []
    optimizer = opt_func(modl.parameters(), lr)
    for epoch in range(num_epochs):
        # Training Phase
        for batch in train_load:
            loss = modl.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evaluate(modl, val_load)
        modl.epoch_end(epoch, result)
        history.append(result)
    return history
