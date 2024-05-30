import sys
import os
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader 
from sklearn.model_selection import KFold # type: ignore
from torch.optim.lr_scheduler import ReduceLROnPlateau # type: ignore

# Add the directory containing your config module to the sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import cfg
from data import data_processing, ds, Augment, Stretch, Amplify
from Conv import Model
import pytest
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    epoch_train_loss = 0
    for batch_id, (x, y_true) in enumerate(train_loader):
        y_pred = model(x.to(device))
        optimizer.zero_grad()
        loss = criterion(y_pred, y_true.long().to(device))
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item() / len(train_loader)
    return epoch_train_loss

def evaluate(model, val_loader, criterion, device):
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for batch_id, (x, y_true) in enumerate(val_loader):
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y_true.long().to(device))
            epoch_val_loss += loss.item() / len(val_loader)
    return epoch_val_loss


def test(model, test_loader, criterion, device):
    model.eval()
    epoch_test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_id, (x, y_true) in enumerate(test_loader):
            y_pred = model(x.to(device))
            loss = criterion(y_pred, y_true.long().to(device))
            epoch_test_loss += loss.item() / len(test_loader)
            _, predicted = torch.max(y_pred.data, 1)
            total += y_true.size(0)
            correct += (predicted == y_true.to(device)).sum().item()

    accuracy = correct / total
    return epoch_test_loss, accuracy

    

def run_experiment():
    # Access hyperparameters directly from cfg
    batch_size = cfg['batch_size']
    epochs = cfg['epochs']
    lr = 1e-3
    criterion = nn.CrossEntropyLoss()
    n_splits = cfg['n_splits']
    patience = cfg['patience']

    epoch_train_losses = []
    epoch_val_losses = []

    # Create data loaders
    train_loader, test_loader, train_df = data_processing()
    augment = Augment([Amplify(), Stretch()])
    kf = KFold(n_splits=n_splits)
    print(f"Starting K-Fold cross-validation with {n_splits} splits")

    for fold_n, (train_idx, val_idx) in enumerate(kf.split(train_df)):
        print(f"Processing fold {fold_n}")
        train_set = ds(train_df.iloc[train_idx, :-1], train_df.iloc[train_idx, -1], transforms=augment)
        val_set = ds(train_df.iloc[val_idx, :-1], train_df.iloc[val_idx, -1])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size * 4)

        model = Model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        lr_sched = ReduceLROnPlateau(optimizer, patience=patience)

        for epoch in range(epochs):
            epoch_train_loss = train(model, train_loader, criterion, optimizer, device)
            epoch_val_loss = evaluate(model, val_loader, criterion, device)

            epoch_train_losses.append(epoch_train_loss)
            epoch_val_losses.append(epoch_val_loss)
            print(f'Fold {fold_n} Epoch {epoch}:\tTrain loss: {epoch_train_loss:0.2e}\tVal loss: {epoch_val_loss:0.2e}\tLR: {optimizer.param_groups[0]["lr"]:0.2e}')

            if lr_sched is None:
                if epoch % 10 == 0 and epoch > 0:
                    optimizer.param_groups[0]['lr'] /= 10
                    print(f'Reducing LR to {optimizer.param_groups[0]["lr"]}')
            else:
                lr_sched.step(epoch_val_loss)

        torch.save(model.state_dict(), f'model_fold_{fold_n}.pth')

if __name__ == "__main__":
    run_experiment()
