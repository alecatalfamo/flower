import logging
from typing import List, Dict
import numpy as np

import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import (
    FitIns, FitRes, EvaluateRes, EvaluateIns,
    Parameters, GetParametersRes, Status, Code,
    ndarrays_to_parameters, parameters_to_ndarrays
)

from fedhomo.train import train_local

logging.basicConfig(level=logging.INFO)


class PlaintextClient(fl.client.Client):
    """Flower client implementing plaintext federated learning."""
    
    def __init__(
        self,
        cid: str,
        trainloader: DataLoader,
        valloader: DataLoader,
        net: torch.nn.Module,
        epochs: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ) -> None:
        """Initialize client with training parameters."""
        logging.info(f"Initializing PlaintextClient for client {cid}")
        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.net = net
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer

    def get_parameters(self, config: Dict[str, str]) -> GetParametersRes:
        """Get current model parameters as Flower Parameters."""
        logging.info(f"Client {self.cid}: Getting parameters")
        try:
            ndarrays = [
                param.cpu().detach().numpy()
                for param in self.net.parameters()
            ]
            parameters = ndarrays_to_parameters(ndarrays)
            
            return GetParametersRes(
                status=Status(Code.OK, "Success"),
                parameters=parameters,
            )
        except Exception as e:
            error_msg = f"Parameter retrieval failed: {str(e)}"
            logging.error(f"Client {self.cid}: {error_msg}")
            return GetParametersRes(
                status=Status(Code.ERROR, error_msg),
                parameters=ndarrays_to_parameters([]),
            )

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set model parameters from given numpy arrays."""
        logging.info(f"Client {self.cid}: Setting model parameters")
        try:
            params_dict = zip(self.net.state_dict().keys(), parameters)
            state_dict = {
                k: torch.from_numpy(v)  # Better dtype preservation
                for k, v in params_dict
            }
            self.net.load_state_dict(state_dict, strict=True)
        except Exception as e:
            error_msg = f"Parameter setting failed: {str(e)}"
            logging.error(f"Client {self.cid}: {error_msg}")
            raise  # Let caller handle this exception

    def fit(self, fit_ins: FitIns) -> FitRes:
        """Train model using received parameters and return updated weights."""
        logging.info(f"Client {self.cid}: Starting training")
        try:
            # Convert parameters to numpy arrays
            parameters = parameters_to_ndarrays(fit_ins.parameters)
            self.set_parameters(parameters)

            # Local training
            train_local(
                self.net, self.trainloader, self.epochs,
                self.criterion, self.optimizer, torch.device("cpu")
            )
            logging.info(f"Client {self.cid}: Training completed")

            # Return updated parameters
            updated_params = [
                param.cpu().detach().numpy()
                for param in self.net.parameters()
            ]
            return FitRes(
                status=Status(Code.OK, "Success"),
                parameters=ndarrays_to_parameters(updated_params),
                num_examples=len(self.trainloader.dataset),
                metrics={},
            )
        except Exception as e:
            error_msg = f"Training failed: {str(e)}"
            logging.error(f"Client {self.cid}: {error_msg}")
            return FitRes(
                status=Status(Code.ERROR, error_msg),
                parameters=None,
                num_examples=0,
                metrics={}
            )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Evaluate model on local validation set."""
        logging.info(f"Client {self.cid}: Evaluating model")
        try:
            loss = self._evaluate_model()
            logging.info(f"Client {self.cid}: Evaluation loss: {loss:.4f}")
            
            return EvaluateRes(
                status=Status(Code.OK, "Success"),
                loss=float(loss),
                num_examples=len(self.valloader.dataset),
                metrics={"loss": loss},
            )
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            logging.error(f"Client {self.cid}: {error_msg}")
            return EvaluateRes(
                status=Status(Code.ERROR, error_msg),
                loss=0.0,
                num_examples=0,
                metrics={}
            )

    def _evaluate_model(self) -> float:
        """Calculate average loss on validation set."""
        self.net.eval()
        total_loss = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for inputs, labels in self.valloader:
                inputs, labels = inputs.to("cpu"), labels.to("cpu")
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                total_samples += inputs.size(0)
        
        return total_loss / total_samples if total_samples > 0 else 0.0