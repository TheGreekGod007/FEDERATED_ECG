import hydra
from omegaconf import DictConfig ,OmegaConf
from torch.utils.data import DataLoader 
from sklearn.model_selection import KFold # type: ignore
from data import data_processing
from data import ds
from data import Augment
from data import Stretch
from data import Amplify
from data import data_processing_federated
import flwr as fl
from Client import generate_client
from server import get_on_fit_config , get_evaluate_fn 
import ray
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Initialize Ray with limited resources
ray.init(num_cpus=4, object_store_memory=1 * 1024 * 1024 * 1024)  # 1GB object store memory


@hydra.main(config_path=".",config_name="base",version_base=None)
def main(cfg: DictConfig):
   

     #2. Prepare dataset 

    train_loader,val_loader,test_loader = data_processing_federated ()
  
      #3. Define Clients 
    client_fn = generate_client(train_loader,val_loader, test_loader , cfg.num_Classes)
    print ("yes")

      #4. Define Strategy 
    client_resources = {"num_cpus": 2, "memory": 4 * 1024 * 1024 * 1024}  # 4 GB in bytes

    strategy = fl.server.strategy.FedAvg(
    fraction_fit=0.00001,
    min_fit_clients=cfg.num_Clients_per_round_fit,
    fraction_evaluate=0.1,
    min_evaluate_clients=cfg.num_Clients_per_round_eval,
    min_available_clients=max(cfg.num_Clients_per_round_fit, cfg.num_Clients_per_round_eval, cfg.num_clients),
    on_fit_config_fn=get_on_fit_config(cfg.config_fit),
    evaluate_fn=get_evaluate_fn(test_loader, cfg.n_splits)
    
)



    print ("strategy executed'")

    # Start Simulation


    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=cfg.num_clients,
        config=fl.server.ServerConfig(num_rounds=cfg.num_round),
        strategy=strategy

    )
    print ("simulation starts")
                                               

      # 6. Save your results   


if __name__=="__main__":

            main ()  
 
    