import logging
import json
from pathlib import Path

from pipeline.download import download_and_extract
from pipeline.ingestion import process_data
from pipeline.loader import create_data_loader
from pipeline.models import ResNetClassifier
from pipeline.training import train_model
from pipeline.testing import test_model

import torch
from torch import nn, optim

def load_config(path: str):
    try:
        with open(path, 'r') as file:
            if path.endswith('.json'):
                return json.load(file)
            else:
                raise ValueError("Unsupported configuration file format.")
    except FileNotFoundError:
        logging.warning(f"Configuration file '{path}' not found. Using default config.")
        return {
            "test_size": 0.2,
            "val_size": 0.2,
            "random_state": 42,
            "lr": 0.001,
            
            "n_batches": 4,
            "batch_indices": [1, 2, 3, 4] ,
        }

def main():
    logging.basicConfig(level=logging.DEBUG)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load configuration
    config_path = "./config.yaml"
    config = load_config(config_path)

    # Step 1: Download data
    cifar10_url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    cifar10_dir = download_and_extract(cifar10_url, "./data/")

    # Step 2: Process data using batch configuration
    train_df, val_df, test_df = process_data(cifar10_dir, config)

    # Step 3: Create data loaders
    train_loader = create_data_loader(train_df, config)
    val_loader = create_data_loader(val_df, config)
    test_loader = create_data_loader(test_df, config)

    # Step 4: Define model, loss function, optimizer
    model = ResNetClassifier(n_classes=10).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    # Step 5: Train model
    best_model_path = train_model(
        model, train_loader, val_loader, loss_function, optimizer,
        num_epochs=config.get("num_epochs", 15), device=device
    )

    # Step 6: Test model
    test_model(model, test_loader, loss_function, device)

if __name__ == "__main__":
    main()
