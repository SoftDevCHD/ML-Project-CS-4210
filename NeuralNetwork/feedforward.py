import matplotlib.pyplot as plt
from classes import *
from utilityfunctions import *
import torch.utils.data
import torch as t
from torch.utils.data import random_split, TensorDataset
from torch.utils.data.dataloader import DataLoader
import numpy as np
import pandas as pd

if __name__ == '__main__':

    dataset = pd.read_csv('international_matches.csv')
    dataframe = dataset.iloc[:, 4:8]
    labelframe = dataset.iloc[:, -1]

    num_frame = dataframe.select_dtypes(include=['int64', 'float64'])

    num_labels = t.Tensor(labelframe.apply(lambda l: encode_label(l)).values)

    input_table = np.array([np.asarray(row) for row in dataframe.itertuples(index=None)])
    input_tensor = torch.tensor(input_table, dtype=torch.float32)
    label_tensor = num_labels
    # F.one_hot(num_labels.to(torch.int64), num_classes=3)
    dataset = TensorDataset(input_tensor, label_tensor)

    val_size = 4000
    train_size = 16000
    test_size = len(dataset) - (val_size + train_size)

    # Divide dataset into 3 unique random subsets
    training_data, validation_data, test_data = random_split(dataset, [train_size, val_size, test_size])

    batch_size = 512

    train_loader = DataLoader(training_data, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(validation_data, batch_size*2, num_workers=4, pin_memory=True)

    # hyper parameters
    item, _ = dataset[0]
    input_size = item.shape[-1]
    hidden_size1 = 10
    hidden_size2 = 10
    num_classes = 3
    epochs = 20
    learning_rate = 0.01

    # Move data from CPU to GPU
    print(torch.cuda.is_available())
    device = get_default_device()

    train_loader = DeviceDataLoader(train_loader, device)
    valid_loader = DeviceDataLoader(valid_loader, device)

    # Create model and move it to currently used device
    model = NeuralNet(input_size, hidden_size1, hidden_size2, num_classes)
    to_device(model, device)

    # Initial evaluation of model
    hist = [evaluate(model, valid_loader)]

    # Training the model
    hist += fit(epochs, learning_rate, model, train_loader, valid_loader)

    with open("traininghistory", "a", newline='') as file:
        file.truncate(0)
        file.write(model.log)

    # Graph loss
    losses = [x['val_loss'] for x in hist]
    epochs = list(range(1, len(losses)+1))
    plt.plot(epochs, losses, '-x')
    for i, j in zip(epochs, losses):
        if i % 2 == 0:
            plt.annotate(str(round(j, 3)), xy=(i, j), xytext=(-12, -12), textcoords='offset points')
    plt.xticks(epochs, [i for i, _ in enumerate(epochs)])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss vs. No. of epochs')
    plt.savefig('LossTrainingModel1')
    plt.clf()

    # Graph accuracy
    accuracies = [x['val_acc'] for x in hist]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig('AccuracyTrainingModel1')

    # Testing model
    x, target = test_data[0]
    print('Label: ', target, ', Predicted: ', predict_winning_team(x, model, device))
