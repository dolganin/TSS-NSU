import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingClassifier


epoch_num = 12
lr = 8e-3
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.linear_stack = torch.nn.Sequential(
            torch.nn.Linear(28*28, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 10)
        )

    def forward(self, x):
        x = self.linear_stack(x)
        return x

def main():
    df = pd.read_csv("mnist_train.csv")
    y = df['label']
    x = df.drop('label', axis=1)
    FC = NeuralNetwork()
    optim = torch.optim.SGD(FC.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for i in range(epoch_num):
        pass



