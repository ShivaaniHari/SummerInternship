import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from typing import cast
import numpy as np
import numpy.random as npr
from torch.utils.data import Subset
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Data owner neural network

import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


# Creating label owner split
def split(nr_clients: int, seed: int) -> list[Subset]:
    rng = npr.default_rng(seed)
    indices = rng.permutation(len(train_dataset))
    splits = np.array_split(indices, nr_clients)

    return [Subset(train_dataset, split) for split in cast(list[list[int]], splits)], indices


class BottomModel(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(BottomModel, self).__init__()
        self.local_out_dim = out_feat  # Final output dimension of the bottom model
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)
        # self.conv1= nn.Conv1d(in_feat, 32, 3, 1)
        # self.conv2 = nn.Conv1d(32, 16, 3, 1)
        self.lin1 = nn.Linear(196, 66)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x: torch.tensor):
        # x= self.conv1(x)
        # x= self.conv2(x)
        x = self.flatten(x)
        x = F.relu(self.lin1(x))
        # x= F.max_pool2d(x, 2)
        x = self.dropout(x)
        return x


# Label owner neural network

class TopModel(nn.Module):
    def __init__(self, local_models, n_outs):
        super(TopModel, self).__init__()
        # top_in_dim= sum([i.local_out_dim for i in local_models])
        self.lin1 = nn.Linear(264, 100)
        self.lin2 = nn.Linear(100, 10)  # Final output = number of possible classes (10 digit types)
        self.act = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        concat_outs = torch.cat(x, dim=1)  # concatenate local model outputs before forward pass
        x = self.act(self.lin1(concat_outs))
        x = F.relu(x)
        x = self.act(self.lin2(x))
        x = self.dropout(x)
        return x


class VFLNetwork(nn.Module):
    def __init__(self, local_models, n_outs):
        super(VFLNetwork, self).__init__()
        self.bottom_models = local_models  # Shared set of bottom models for the entire process of training
        self.top_models = [TopModel(self.bottom_models, n_outs) for _ in
                           range(5)]  # Creating 5 top models, one for each label owner
        self.optimizers = [optim.AdamW(self.top_models[i].parameters()) for i in range(5)]

        self.criterion = nn.CrossEntropyLoss()
        self.valid_owner = [0, 1, 2, 3, 4]
        self.indices = [0] * 5

    # Need to change the nature of x as well, it is going to be a list of lists (label as well as data partitioning done)

    def train_with_settings(self, epochs, batch_sz, x, y):
        num_batches = (12000 // batch_sz) * 5 if 12000 % batch_sz == 0 else (12000 // batch_sz + 1) * 5
        for epoch in range(epochs):
            for opt in self.optimizers:
                opt.zero_grad()
            total_loss = 0.0
            correct = 0.0
            total = 0.0
            for j in range(10000):
                label_owner_id = npr.choice(self.valid_owner)
                curr_data = x[label_owner_id]
                curr_labels = y[label_owner_id]
                x_minibatch = [p[int(self.indices[label_owner_id]):int(self.indices[label_owner_id] + batch_sz)] for p
                               in curr_data]
                y_minibatch = torch.tensor(
                    curr_labels[int(self.indices[label_owner_id]):int(self.indices[label_owner_id] + batch_sz)],
                    dtype=torch.long)
                self.indices[label_owner_id] += batch_sz
                if self.indices[label_owner_id] == 12000:
                    self.valid_owner.remove(label_owner_id)

                outs = self.forward(x_minibatch, label_owner_id)
                pred = torch.argmax(outs, dim=1)
                actual = y_minibatch
                correct += torch.sum((pred == actual))
                total += len(actual)
                loss = self.criterion(outs, y_minibatch)
                if loss == torch.nan:
                    print("HERE")
                total_loss += loss
                loss.backward()
                self.optimizers[label_owner_id].step()

            print(
                f"Epoch: {epoch} Train accuracy: {correct * 100 / total:.2f}% Loss: {total_loss.detach().numpy() / num_batches:.3f}")

            if (epoch + 1) % 25 == 0:
                self.aggregation()
            # accuracy_at_each_epoch.append(total_loss.detach().numpy()/num_batches)
            # if epoch== epochs-1:
            #     training_loss.append(total_loss.detach().numpy()/num_batches)

    def forward(self, x, label_owner_id):
        local_outs = [self.bottom_models[i](x[i]) for i in range(len(self.bottom_models))]
        return self.top_models[label_owner_id](local_outs)

    def test(self, x, y, label_owner_id):  # Additional parameter to define which label owner's model is to be tested.
        with torch.no_grad():
            outs = self.forward(x, label_owner_id)
            pred = torch.argmax(outs, dim=1)
            actual = torch.tensor(y)
            accuracy = torch.sum((pred == actual)) / len(actual)
            loss = self.criterion(outs, actual)
            return accuracy, loss

    def aggregation(self):
        parameter_set = []
        avg_parameters = OrderedDict()
        with torch.no_grad():
            for i in range(5):
                parameter_set.append(self.top_models[i].state_dict())

            for key in parameter_set[0]:
                avg_parameters[key] = (parameter_set[0][key] + parameter_set[1][key] + parameter_set[2][key] +
                                       parameter_set[3][key] + parameter_set[4]) / 5

            for i in range(5):
                self.top_models[i].load_state_dict(avg_parameters)


if __name__ == "__main__":
    data_path_str = "./data"
    ETA = "\N{GREEK SMALL LETTER ETA}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.deterministic = True

    transform = transforms.Compose([
        transforms.ToTensor(),
        # normalize by training set mean and standard deviation
        #  resulting data has mean=0 and std=1
        # transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = MNIST(data_path_str, train=True, download=False, transform=transform)
    test_dataset = MNIST(data_path_str, train=False, download=False, transform=transform)
    test_loader = DataLoader(
        MNIST(data_path_str, train=False, download=False, transform=transform),
        # decrease batch size if running into memory issues when testing
        # a bespoke generator is passed to avoid reproducibility issues
        shuffle=False, drop_last=False, batch_size=10000, generator=torch.Generator())

    # Partitioning data (each image into 4 parts)

    data1 = torch.stack([a[0:7] / 255 for a in train_dataset.data])
    data2 = torch.stack([a[7:14] / 255 for a in train_dataset.data])
    data3 = torch.stack([a[14:21] / 255 for a in train_dataset.data])
    data4 = torch.stack([a[21:28] / 255 for a in train_dataset.data])

    # Test dataset
    # Need to partition this as well for testing each of the client models after training

    test1 = torch.stack([a[0:7] / 255 for a in test_dataset.data])
    test2 = torch.stack([a[7:14] / 255 for a in test_dataset.data])
    test3 = torch.stack([a[21:28] / 255 for a in test_dataset.data])
    test4 = torch.stack([a[21:28] / 255 for a in test_dataset.data])
    test_labels = [test_loader.dataset[i][1] for i in range(len(test_loader.dataset))]

    # Creating label split
    # Sample_split contains the labels after permuting the original label set
    # Sample_ids contains the permutation used for the randomization process

    sample_split, sample_ids = split(5, 42)

    label_owner1 = sample_split[0]
    label_owner2 = sample_split[1]
    label_owner3 = sample_split[2]
    label_owner4 = sample_split[3]
    label_owner5 = sample_split[4]

    label_id1 = sample_ids[0:12000]
    label_id2 = sample_ids[12000:24000]
    label_id3 = sample_ids[24000:36000]
    label_id4 = sample_ids[36000:48000]
    label_id5 = sample_ids[48000:60000]

    # Aligning the data across each of the owners and label owner 1
    # Retrieving data corresponding to which labels are with label owner 1

    labels1 = [label_owner1[i][1] for i in range(len(label_owner1))]
    dataA_label1 = torch.stack([data1[i] for i in label_id1])
    dataB_label1 = torch.stack([data2[i] for i in label_id1])
    dataC_label1 = torch.stack([data3[i] for i in label_id1])
    dataD_label1 = torch.stack([data4[i] for i in label_id1])
    data_labels1 = [dataA_label1, dataB_label1, dataC_label1, dataD_label1]

    # Doing the same for each of the other 4 label owners
    labels2 = [label_owner2[i][1] for i in range(len(label_owner2))]
    dataA_label2 = torch.stack([data1[i] for i in label_id2])
    dataB_label2 = torch.stack([data2[i] for i in label_id2])
    dataC_label2 = torch.stack([data3[i] for i in label_id2])
    dataD_label2 = torch.stack([data4[i] for i in label_id2])
    data_labels2 = [dataA_label2, dataB_label2, dataC_label2, dataD_label2]

    labels3 = [label_owner3[i][1] for i in range(len(label_owner3))]
    dataA_label3 = torch.stack([data1[i] for i in label_id3])
    dataB_label3 = torch.stack([data2[i] for i in label_id3])
    dataC_label3 = torch.stack([data3[i] for i in label_id3])
    dataD_label3 = torch.stack([data4[i] for i in label_id3])
    data_labels3 = [dataA_label3, dataB_label3, dataC_label3, dataD_label3]

    labels4 = [label_owner4[i][1] for i in range(len(label_owner4))]
    dataA_label4 = torch.stack([data1[i] for i in label_id4])
    dataB_label4 = torch.stack([data2[i] for i in label_id4])
    dataC_label4 = torch.stack([data3[i] for i in label_id4])
    dataD_label4 = torch.stack([data4[i] for i in label_id4])
    data_labels4 = [dataA_label4, dataB_label4, dataC_label4, dataD_label4]

    labels5 = [label_owner1[i][1] for i in range(len(label_owner5))]
    dataA_label5 = torch.stack([data1[i] for i in label_id5])
    dataB_label5 = torch.stack([data2[i] for i in label_id5])
    dataC_label5 = torch.stack([data3[i] for i in label_id5])
    dataD_label5 = torch.stack([data4[i] for i in label_id5])
    data_labels5 = [dataA_label5, dataB_label5, dataC_label5, dataD_label5]

    accuracy_at_each_epoch = []

    # training_loss= []
    # test_loss= []
    # epoch_nums= []

    EPOCHS = 500
    BATCH_SIZE = 64
    bottom_models = [BottomModel(7, 32)] * 4
    final_out_dims = 10

    Network = VFLNetwork(bottom_models, final_out_dims)
    datasets_with_splits = [[dataA_label1, dataB_label1, dataC_label1, dataD_label1],
                            [dataA_label2, dataB_label2, dataC_label2, dataD_label2],
                            [dataA_label3, dataB_label3, dataC_label3, dataD_label3],
                            [dataA_label4, dataB_label4, dataC_label4, dataD_label4],
                            [dataA_label5, dataB_label5, dataC_label5, dataD_label5]]
    label_set_split = [labels1, labels2, labels3, labels4, labels5]
    Network.train_with_settings(EPOCHS, BATCH_SIZE, datasets_with_splits, label_set_split)
