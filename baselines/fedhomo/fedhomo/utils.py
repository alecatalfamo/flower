import logging
from pathlib import Path
from typing import Any, List, Union, Tuple, Optional, Dict
from functools import reduce
import numpy as np
import tenseal as ts

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HomomorphicError(Exception):
    """Base exception for homomorphic operations"""
    pass

class EncryptionError(HomomorphicError):
    """Exception raised for encryption failures"""
    pass

class DecryptionError(HomomorphicError):
    """Exception raised for decryption failures"""
    pass

def validate_context(context: ts.Context) -> None:
    """Validate TenSEAL context meets security requirements"""
    if not context.is_public():
        raise ValueError("Context must be public for aggregation")
    if "CKKS" not in context.scheme().name.upper():
        raise ValueError("Only CKKS homomorphic scheme supported")

def serialize_to_bytes(data: Union[np.ndarray, List]) -> List[bytes]:
    """Recursively serialize array-like data to bytes with validation"""
    if not isinstance(data, (list, np.ndarray)):
        raise TypeError("Input must be array-like (list or numpy array)")
    
    if isinstance(data, np.ndarray):
        return [data.tobytes()]
    
    return [serialize_to_bytes(item) for item in data]

def encrypt_ndarray(
    array: np.ndarray,
    context: ts.Context
) -> ts.CKKSVector:
    """Encrypt numpy array with context validation"""
    validate_context(context)
    try:
        return ts.ckks_vector(context, array)
    except Exception as e:
        logger.error(f"Encryption failed: {str(e)}")
        raise EncryptionError("Array encryption failed") from e

def decrypt_vector(
    encrypted_data: bytes,
    context: ts.Context
) -> np.ndarray:
    """Decrypt single encrypted vector"""
    try:
        vec = ts.ckks_vector_from(context, encrypted_data)
        vec.link_context(context)
        return np.array(vec.decrypt())
    except Exception as e:
        logger.error(f"Decryption failed: {str(e)}")
        raise DecryptionError("Vector decryption failed") from e

class EncryptedAggregator:
    """Handles secure aggregation of encrypted model updates"""
    
    def __init__(self, context: ts.Context):
        validate_context(context)
        self.context = context
    
    def weighted_sum(
        self,
        updates: List[Tuple[List[ts.CKKSVector], int]]
    ) -> List[ts.CKKSVector]:
        """Compute weighted sum of encrypted updates"""
        total_weight = sum(weight for _, weight in updates)
        weighted_vectors = []
        
        for vectors, weight in updates:
            scaled_weight = weight / total_weight
            weighted_vectors.append([v * scaled_weight for v in vectors])
            
        return [reduce(lambda a, b: a + b, layer) for layer in zip(*weighted_vectors)]

    def serialize_vectors(
        self,
        vectors: List[ts.CKKSVector]
    ) -> List[bytes]:
        """Serialize encrypted vectors for transmission"""
        return [v.serialize() for v in vectors]

class ModelInspector:
    """Validates and inspects encrypted models"""
    
    @staticmethod
    def encryption_status(
        model: Any,
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """Analyze encryption status of model structure"""
        def inspect(obj, depth):
            if depth > max_depth:
                return {"type": "max_depth_reached"}
                
            if isinstance(obj, ts.CKKSVector):
                return {
                    "type": "CKKSVector",
                    "is_encrypted": True,
                    "size": len(obj)
                }
                
            if isinstance(obj, (list, np.ndarray)):
                sample = obj[min(2, len(obj)-1)] if len(obj) > 0 else None
                return {
                    "type": type(obj).__name__,
                    "length": len(obj),
                    "sample": inspect(sample, depth+1)
                }
                
            return {
                "type": type(obj).__name__,
                "is_encrypted": False
            }
            
        return inspect(model, 0)

    @staticmethod
    def count_encrypted_layers(model: Any) -> int:
        """Count encrypted layers in nested structure"""
        if isinstance(model, ts.CKKSVector):
            return 1
        if isinstance(model, (list, np.ndarray)):
            return sum(ModelInspector.count_encrypted_layers(item) for item in model)
        return 0

class SecureSerializer:
    """Handles safe conversion between data formats"""
    
    @staticmethod
    def numpy_to_serializable(arr: np.ndarray) -> list:
        """Convert numpy array to JSON-serializable format"""
        return arr.tolist()
    
    @staticmethod
    def encrypted_to_bytes(vectors: List[ts.CKKSVector]) -> List[bytes]:
        """Serialize encrypted vectors while preserving structure"""
        return [v.serialize() for v in vectors]

class HomomorphicClientHandler:
    """Handles all homomorphic encryption operations for a client"""
    
    def __init__(self, cid: str):
        self.cid = cid
        self.logger = logging.getLogger(f"CryptoHandler-{cid}")
        self.public_context, self.secret_context = self._load_security_contexts()
        self._validate_contexts()

    def _load_security_contexts(self) -> Tuple[ts.Context, ts.Context]:
        """Load and validate encryption contexts"""
        try:
            # In production, load from secure storage
            with open("public_context.pkl", "rb") as f:
                public_ctx = ts.context_from(f.read())
            
            with open("secret_context.pkl", "rb") as f:
                secret_ctx = ts.context_from(f.read())
            
            return public_ctx, secret_ctx
        except Exception as e:
            self.logger.error(f"Context loading failed: {str(e)}")
            raise EncryptionError("Security context initialization failed") from e

    def _validate_contexts(self) -> None:
        """Validate loaded contexts meet security requirements"""
        if not self.public_context.is_public():
            raise EncryptionError("Public context must be public")
        if self.secret_context.is_public():
            raise EncryptionError("Secret context must be private")
        if "CKKS" not in self.public_context.scheme().name.upper():
            raise EncryptionError("Only CKKS scheme supported")

    def encrypt_parameters(self, ndarrays: List[np.ndarray]) -> List[bytes]:
        """Encrypt model parameters using CKKS scheme"""
        try:
            encrypted_params = []
            for arr in ndarrays:
                vec = ts.ckks_vector(self.public_context, arr)
                encrypted_params.append(vec.serialize())
            return encrypted_params
        except Exception as e:
            self.logger.error(f"Encryption failed: {str(e)}")
            raise EncryptionError("Parameter encryption failed") from e

    def decrypt_parameters(self, encrypted_data: List[bytes]) -> List[np.ndarray]:
        """Decrypt received parameters using secret context"""
        try:
            decrypted_params = []
            for data in encrypted_data:
                vec = ts.ckks_vector_from(self.secret_context, data)
                vec.link_context(self.secret_context)
                decrypted_params.append(np.array(vec.decrypt()))
            return decrypted_params
        except Exception as e:
            self.logger.error(f"Decryption failed: {str(e)}")
            raise DecryptionError("Parameter decryption failed") from e

    def process_incoming_parameters(self, parameters: List[np.ndarray]) -> List[np.ndarray]:
        """Validate and deserialize incoming parameters"""
        if not isinstance(parameters, list) or len(parameters) == 0:
            raise DecryptionError("Invalid parameters format")
        
        try:
            # Convert numpy arrays to bytes
            byte_params = [param.tobytes() for param in parameters]
            return self.decrypt_parameters(byte_params)
        except Exception as e:
            self.logger.error(f"Parameter processing failed: {str(e)}")
            raise DecryptionError("Parameter deserialization failed") from e