import flwr as fl
import torch
import logging
import pandas as pd
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from omegaconf import DictConfig
from Conv import Model  # Import your model definition
from data import ds  # Ensure you have the dataset class imported

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from omegaconf import DictConfig
def get_on_fit_config(config: DictConfig):
    def fit_config_fn(server_round: int):
        return {
            'lr': config.lr,
            'momentum': config.momentum,
            'local_epochs': config.local_epochs
        }
    return fit_config_fn

def get_evaluate_fn(test_loader, n_splits):
    def evaluate_fn(server_round: int, parameters, config):
        model = Model().to(device)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load parameters into the model
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        

        predictions = []

        model.eval()
        with torch.no_grad():
            for fold in range(n_splits):
                model = Model().to(device)
                model.load_state_dict(torch.load(f'model_fold_{fold}.pth'))
                model.eval()
                
                fold_predictions = []
                labels = []
                for batch_id, (x, y_true) in enumerate(test_loader):
                    y_pred = model(x.to(device)).argmax(1)
                    fold_predictions.extend(y_pred.squeeze().tolist())
                    labels.extend(y_true.squeeze().tolist())
                predictions.append(fold_predictions)
            
        predictions.append(labels)
        all_prediction_labels = pd.DataFrame(
            np.array(predictions).T, 
            columns=[f'fold {n}' for n in range(n_splits)] + ['Label']
        )

        final_predictions = all_prediction_labels.iloc[:, :-1].mode(1).iloc[:, 0]
        prediction_labels = pd.DataFrame(
            np.hstack([
                final_predictions.values.reshape(-1, 1), 
                all_prediction_labels.loc[:, 'Label'].values.reshape(-1, 1)]
            ),
            columns=['Prediction', 'Label']
        )
        
        cm = prediction_labels.groupby(['Prediction', 'Label']).size().unstack(0)
        logger.info(f"Confusion Matrix: \n{cm}")

        accuracy = (prediction_labels['Prediction'] == prediction_labels['Label']).mean()
        loss = 0.0  # You can define a loss function if required

        return loss, len(test_loader.dataset), {"accuracy": accuracy}

    return evaluate_fn

if __name__ == "__main__":
    # Assuming you have a config object, adjust as needed
    config = DictConfig({
        "lr": 0.001,
        "momentum": 0.9,
        "local_epochs": 5
    })

 