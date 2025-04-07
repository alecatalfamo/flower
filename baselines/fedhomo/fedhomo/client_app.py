"""fedhomo: A Flower Baseline."""

import torch
import logging

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fedhomo.dataset import load_data
from fedhomo.model import Net, get_weights, set_weights, test, train
from fedhomo.client import PlaintextClient
from fedhomo.encrypted_client import EncryptedFlowerClient

logging.basicConfig(level=logging.INFO)


def client_fn(context: Context):
    """Construct a Client that will be run in a ClientApp."""
    logging.info(f"Client {context.cid}: Loading data for partition {partition_id} of {num_partitions}")
    # Load model and data
    net = Net()
    partition_id = int(context.node_config["partition-id"])
    num_partitions = int(context.node_config["num-partitions"])
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]    
    # Create client
    if False:
        client = EncryptedFlowerClient(net, trainloader, valloader, local_epochs).to_client()
    else:
        client = PlaintextClient(net, trainloader, valloader, local_epochs).to_client()

    # Return Client instance
    return PlaintextClient(
        cid=context.cid,
        trainloader=trainloader,
        valloader=valloader,
        net=net,
        epochs=local_epochs,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9),
    )


# Flower ClientApp
app = ClientApp(client_fn)
