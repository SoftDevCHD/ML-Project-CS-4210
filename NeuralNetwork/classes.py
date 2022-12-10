import torch.nn as nn
import torch.nn.functional as F
from utilityfunctions import *


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""
    def __init__(self, dl, dev):
        self.dl = dl
        self.device = dev

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


class ClassificationBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.log = 'Creating base model'

    def training_step(self, batch):
        data, labels = batch
        out = self(data)  # Generate predictions
        loss = F.cross_entropy(out, labels.long())  # Calculate loss
        return loss

    def validation_step(self, batch):
        data, labels = batch
        out = self(data)  # Generate predictions
        loss = F.cross_entropy(out, labels.long())  # Calculate loss
        acc = accuracy(out, labels)  # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.3f}, val_acc: {:.3f}".format(epoch+1, result['val_loss'], result['val_acc']))
        self.log += "Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}\n".format(epoch+1, result['val_loss'], result['val_acc'])


class NeuralNet(ClassificationBase):
    def __init__(self, in_size, hid_size1, hid_size2, out_size):
        super().__init__()
        # hidden layer 1
        self.linear1 = nn.Linear(in_size, hid_size1)
        # hidden layer 2
        self.linear2 = nn.Linear(hid_size1, hid_size2)
        # output layer
        self.linear3 = nn.Linear(hid_size2, out_size)

    def forward(self, x):
        # Get intermediate outputs using hidden layer 1
        out = self.linear1(x)

        # Apply activation function
        out = F.relu(out)

        # Get next intermediate outputs using hidden layer 2
        out = self.linear2(out)

        # Apply activation function
        out = F.softmax(out, dim=1)

        # Get predictions using output layer
        out = self.linear3(out)
        return out
