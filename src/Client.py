from typing import Dict
from chex import Scalar
import flwr as fl
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import data_processing, Augment, Amplify, Stretch, ds, data_processing_federated  # Import your modules
import hydra
from config import cfg
from collections import OrderedDict
from Conv import Model
from omegaconf import DictConfig ,OmegaConf
import logging
from Centralized import train, evaluate ,test

from typing import Dict, Union , List 
import numpy as np

logger = logging.getLogger(__name__)



trainloader, valloader, testloader = data_processing_federated()

print("Now federated")



class FlowerClient(fl.client.NumPyClient):
    def __init__(self, trainloader,valloader , testloader ) -> None:
        super().__init__()
        self.trainloader = trainloader
        self.valloader = valloader
        self.testloader = testloader
        self.model = Model()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       # self.num_classes = num_classes

    def get_parameters(self, config: Dict[str, Scalar]):
         print ("get parameter called ")
         logger.info("Client: get_parameters called")
         return [val.cpu().numpy() for val in self.model.state_dict().values()]
         

    def set_parameters(self, parameters):
        print ("set parameters called ")
        logger.info("Client: set_parameters called")
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict}) 
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config): 
        model = self.model.to(self.device)
        # copy parameters sent by the server into client's local model
        self.set_parameters(parameters)
        lr = config['lr']
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        lr_sched = ReduceLROnPlateau(optimizer, patience=config['patience'])
        epochs = cfg['local_epoch']
        epoch_train_losses = []
        epoch_val_losses = []
        # do local training
        for fold_n, (train_loader, val_loader) in enumerate(zip(self.trainloader, self.valloader)):
            print(f"Training on fold {fold_n}")
            for epoch in range(epochs):
                epoch_train_loss = train(self.model, train_loader, criterion, optimizer, self.device)
                epoch_val_loss = evaluate(self.model, val_loader, criterion, self.device)

                epoch_train_losses.append(epoch_train_loss)
                epoch_val_losses.append(epoch_val_loss)
                print(f'Fold {fold_n} Epoch {epoch}:\tTrain loss: {epoch_train_loss:.2e}\tVal loss: {epoch_val_loss:.2e}\tLR: {optimizer.param_groups[0]["lr"]:.2e}')

                if lr_sched is None:
                    if epoch % 10 == 0 and epoch > 0:
                        optimizer.param_groups[0]['lr'] /= 10
                        print(f'Reducing LR to {optimizer.param_groups[0]["lr"]}')
                else:
                    lr_sched.step(epoch_val_loss)

            torch.save(model.state_dict(), f'model_fold_{fold_n}.pth')
        return self.get_parameters({}), len(self.trainloader.dataset), {}


    def evaluate (self ,parameters:np.ndarray,config:Dict[str,Scalar]): 
        print ("evaluate function called ")
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        epoch_test_loss, accuracy=   test(self.model , self.testloader, criterion,  self.device)
        return float(epoch_test_loss) , len(self.testloader) , {'accuracy' : accuracy}
    

def generate_client(trainloaders, valloaders, testloaders,num_classes):
    def client_fn (cid:str) : 
        print ("generate client function called")
        return FlowerClient(trainloader=trainloaders[int(cid)],
                                valloader= valloaders[int(cid)],
                                testloader= trainloaders[int(cid)],
                                num_classes = num_classes)
        
    return client_fn