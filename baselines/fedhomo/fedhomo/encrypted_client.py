import logging
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
import flwr as fl
from flwr.common import (
    FitIns, FitRes, EvaluateRes, EvaluateIns,
    Parameters, GetParametersRes, Status, Code,
    ndarrays_to_parameters, parameters_to_ndarrays
)
import tenseal as ts
import numpy as np

from fedhomo.train import train_local
from fedhomo.utils import (
    HomomorphicError,
    EncryptionError,
    DecryptionError,
    validate_context,
    HomomorphicClientHandler
)

logging.basicConfig(level=logging.INFO)

class EncryptedFlowerClient(fl.client.Client):
    """Federated learning client with certified homomorphic encryption support."""
    
    def __init__(
        self,
        cid: str,
        trainloader: DataLoader,
        valloader: DataLoader,
        net: torch.nn.Module,
        epochs: int,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer
    ):
        self.cid = cid
        self.trainloader = trainloader
        self.valloader = valloader
        self.net = net
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.logger = logging.getLogger(f"EncryptedClient-{cid}")
        self.crypto_handler = HomomorphicClientHandler(cid)

    def get_parameters(self, config: Dict[str, str]) -> GetParametersRes:
        """Secure parameter extraction with homomorphic encryption"""
        try:
            ndarrays = [param.cpu().detach().numpy() for param in self.net.parameters()]
            encrypted_params = self.crypto_handler.encrypt_parameters(ndarrays)
            return GetParametersRes(
                status=Status(Code.OK, "Success"),
                parameters=ndarrays_to_parameters(encrypted_params),
            )
        except EncryptionError as e:
            return self._error_response(f"Encryption failed: {str(e)}")

    def set_parameters(self, parameters: Parameters) -> None:
        """Secure parameter application with validation"""
        try:
            encrypted_ndarrays = parameters_to_ndarrays(parameters)
            decrypted_params = self.crypto_handler.process_incoming_parameters(encrypted_ndarrays)
            
            # Update model weights
            state_dict = {
                k: torch.from_numpy(v) 
                for k, v in zip(self.net.state_dict().keys(), decrypted_params)
            }
            self.net.load_state_dict(state_dict, strict=True)
        except DecryptionError as e:
            raise RuntimeError(f"Decryption error: {str(e)}") from e

    def fit(self, fit_ins: FitIns) -> FitRes:
        """Secure training cycle with encrypted parameter processing"""
        try:
            # Process and decrypt parameters
            encrypted_params = parameters_to_ndarrays(fit_ins.parameters)
            decrypted = self.crypto_handler.process_incoming_parameters(encrypted_params)
            self.set_parameters(ndarrays_to_parameters(decrypted))

            # Local training
            train_local(
                self.net, self.trainloader, self.epochs,
                self.criterion, self.optimizer, torch.device("cpu")
            )

            # Encrypt updated parameters
            updated_params = [param.cpu().detach().numpy() for param in self.net.parameters()]
            encrypted_updates = self.crypto_handler.encrypt_parameters(updated_params)
            
            return FitRes(
                status=Status(Code.OK, "Success"),
                parameters=ndarrays_to_parameters(encrypted_updates),
                num_examples=len(self.trainloader.dataset),
                metrics={"client": self.cid},
            )
        except HomomorphicError as e:
            return self._error_response(f"Security error: {str(e)}")

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """Secure model evaluation"""
        try:
            loss = self._calculate_validation_loss()
            return EvaluateRes(
                status=Status(Code.OK, "Success"),
                loss=float(loss),
                num_examples=len(self.valloader.dataset),
                metrics={"loss": loss, "client": self.cid},
            )
        except Exception as e:
            return self._error_response(f"Evaluation failed: {str(e)}")

    def _calculate_validation_loss(self) -> float:
        """Compute validation loss"""
        self.net.eval()
        total_loss = 0.0
        with torch.no_grad():
            for inputs, labels in self.valloader:
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
        return total_loss / len(self.valloader.dataset)

    def _error_response(self, message: str) -> GetParametersRes:
        """Generate standardized error response"""
        self.logger.error(message)
        return GetParametersRes(
            status=Status(Code.ERROR, message),
            parameters=Parameters(tensors=[], tensor_type=""),
        )