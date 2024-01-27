from torch import nn
import torch
import data_process


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()

        self.linear_relu1 = nn.Linear(input_size, 32)
        self.linear_relu2 = nn.Linear(32, 64)
        self.linear3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.linear_relu1(x)
        x = nn.functional.relu(x)

        x = self.linear_relu2(x)
        x = nn.functional.relu(x)

        x = self.linear3(x)
        return x


    def train_model(model, train_dataloader, criterion, optimizer, device):
        model.train()
        total_loss = 0
        for _, (x, y) in enumerate(train_dataloader):
            x = x.to(device)
            y = y.to(device)
            y_pred = model(x)
            loss = criterion(y_pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss
        return total_loss.data


    def evaluate_model(model, test_dataloader, criterion, device):
        model.eval()
        total_loss = 0
        for _, (x, y) in enumerate(test_dataloader):
            with torch.no_grad():
                x = x.to(device)
                y = y.to(device)

                y_pred = model(x)
                loss = criterion(y_pred, y)
                total_loss += loss
        return loss.data