import torch
import flwr as fl
from typing import List, Tuple, Dict, Optional, Any
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

import tenseal as ts
import time
import numpy as np
import logging
from pathlib import Path

from fedhomo.utils import (
    HomomorphicError,
    EncryptionError,
    DecryptionError,
    validate_context,
    EncryptedAggregator,
    ModelInspector,
    SecureSerializer
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("homomorphic_fedavg.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HomomorphicFedAvg")

class HomomorphicFedAvg(fl.server.strategy.FedAvg):
    """Federated Averaging strategy with certified homomorphic encryption support"""
    
    def __init__(self, accept_failures: bool = True):
        """Initialize strategy with security context"""
        super().__init__()
        self.accept_failures = accept_failures
        self.aggregator = self._initialize_aggregator()
        self.inspector = ModelInspector()
        self.serializer = SecureSerializer()
        logger.info("Initialized HomomorphicFedAvg strategy with security components")

    def _initialize_aggregator(self) -> EncryptedAggregator:
        """Load and validate encryption context"""
        try:
            context = self._load_public_context()
            return EncryptedAggregator(context)
        except Exception as e:
            logger.critical(f"Failed to initialize aggregator: {str(e)}")
            raise

    def _load_public_context(self) -> ts.Context:
        """Load and validate public encryption context"""
        try:
            context_path = Path("public_context.pkl")
            with context_path.open("rb") as f:
                context = ts.context_from(f.read())
            validate_context(context)
            logger.info("Loaded validated encryption context")
            return context
        except FileNotFoundError as e:
            logger.error(f"Missing context file: {e.filename}")
            raise
        except Exception as e:
            logger.error(f"Context validation failed: {str(e)}")
            raise

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Secure aggregation pipeline with homomorphic encryption"""
        start_time = time.time()
        metrics = {
            "server_round": server_round,
            "clients_processed": len(results),
            "errors": 0
        }

        try:
            # Validate preconditions
            if not self._validate_aggregation_preconditions(failures, results):
                return None, metrics

            # Process client updates
            client_updates = self._process_client_updates(results, metrics)

            # Perform homomorphic aggregation
            aggregated_vectors = self.aggregator.weighted_sum(client_updates)
            
            # Validate aggregated model
            self._validate_aggregated_model(aggregated_vectors, metrics)

            # Serialize for distribution
            serialized = self.serializer.encrypted_to_bytes(aggregated_vectors)
            parameters = ndarrays_to_parameters(
                [np.frombuffer(vec, dtype=np.uint8) for vec in serialized]
            )

            # Collect performance metrics
            metrics.update({
                "aggregation_time": time.time() - start_time,
                "encrypted_layers": len(aggregated_vectors),
                "encrypted_status": self.inspector.encryption_status(aggregated_vectors)
            })

            return parameters, metrics

        except HomomorphicError as e:
            logger.error(f"Security critical error: {str(e)}")
            metrics.update({"error": str(e), "errors": metrics["errors"]+1})
            return None, metrics
        except Exception as e:
            logger.error(f"Unexpected aggregation error: {str(e)}")
            metrics.update({"error": str(e), "errors": metrics["errors"]+1})
            return None, metrics

    def _validate_aggregation_preconditions(
        self,
        failures: List[BaseException],
        results: List[Tuple[ClientProxy, FitRes]]
    ) -> bool:
        """Validate conditions for safe aggregation"""
        if failures and not self.accept_failures:
            logger.error(f"Critical failures detected: {len(failures)}")
            return False
            
        if not results:
            logger.warning("No results to aggregate")
            return False
            
        return True

    def _process_client_updates(
        self,
        results: List[Tuple[ClientProxy, FitRes]],
        metrics: Dict[str, Any]
    ) -> List[Tuple[List[ts.CKKSVector], int]]:
        """Process and validate client updates"""
        client_updates = []
        
        for client, res in results:
            try:
                encrypted_params = parameters_to_ndarrays(res.parameters)
                vectors = self.aggregator.process_client_update(encrypted_params)
                
                # Validate encryption status
                encryption_status = self.inspector.encryption_status(vectors)
                if not self._validate_encryption(encryption_status):
                    raise EncryptionError("Invalid encryption format")
                
                client_updates.append((vectors, res.num_examples))
                logger.info(f"Processed update from {client.cid}")
                
            except (EncryptionError, DecryptionError) as e:
                logger.warning(f"Invalid update from {client.cid}: {str(e)}")
                metrics["errors"] += 1
                if not self.accept_failures:
                    raise
            except Exception as e:
                logger.error(f"Unexpected error processing {client.cid}: {str(e)}")
                metrics["errors"] += 1
                if not self.accept_failures:
                    raise

        return client_updates

    def _validate_encryption(self, status: Dict[str, Any]) -> bool:
        """Validate encryption structure meets requirements"""
        if status.get("is_encrypted", False) is not True:
            logger.error("Unencrypted parameters detected")
            return False
            
        if status.get("type") != "CKKSVector":
            logger.error("Invalid encryption type detected")
            return False
            
        return True

    def _validate_aggregated_model(
        self,
        model: List[ts.CKKSVector],
        metrics: Dict[str, Any]
    ) -> None:
        """Post-aggregation validation checks"""
        encrypted_count = self.inspector.count_encrypted_layers(model)
        metrics["encrypted_layers"] = encrypted_count
        
        if encrypted_count == 0:
            raise EncryptionError("Aggregation produced unencrypted model")
            
        logger.info(f"Validated aggregated model with {encrypted_count} encrypted layers")