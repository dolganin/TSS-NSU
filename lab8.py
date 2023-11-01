import pandas as pd
import torch
import torchmetrics.classification
from sklearn.ensemble import GradientBoostingClassifier
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn import model_selection
from sklearn.metrics import accuracy_score


epoch_num = 1
lr = 8e-3
batch_size = 64

class MNISTDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = np.array(self.data.iloc[idx, 1:], dtype=np.uint8).reshape((28, 28))
        label = np.array(self.data.iloc[idx, 0])
        return image, label
class NeuralNetwork(torch.nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flat = torch.nn.Flatten()
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
        x = self.flat(x)
        x = x.to(torch.float32)
        x = self.linear_stack(x)
        return x

def main():
    train_dataset = MNISTDataset("mnist_train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = MNISTDataset("mnist_test.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    FC = NeuralNetwork()
    optim = torch.optim.SGD(FC.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_score = torchmetrics.classification.MulticlassAccuracy(num_classes=10)

    for i in range(epoch_num):
        for images, labels in train_dataloader:
            FC.train()
            optim.zero_grad()
            target = FC(images)
            loss = loss_fn(target, labels)
            loss.backward()
            optim.step()

    test_acc = []
    for images, labels in test_dataloader:
        FC.eval()
        optim.zero_grad()
        target = FC(images)

        test_acc.append(acc_score(target, labels).numpy())

    print("Accuracy for testing NN = " + str(np.array(test_acc).mean()))

    df = pd.read_csv("diabetes.csv")
    y = df["Outcome"]
    x = df.drop("Outcome", axis=1)

    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.2, random_state=42)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)
    model.fit(X_train, y_train)
    predict = model.predict(X_test)
    print("Accuracy for testing GB = " + str(accuracy_score(y_test, predict)))

main()


