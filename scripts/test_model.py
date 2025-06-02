#!/usr/bin/env python3
import logging
import sys
import os
import yaml
import pandas as pd

import torch
from torch import nn

from pipeline.models import ResNetClassifier
from pipeline.loader import create_data_loader
from pipeline.testing import test_model


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml", "r"))
    prepare_cfg = params["prepare"]
    train_cfg = params["train"]
    test_cfg = params["test"]

    test_pkl = prepare_cfg["test_split_path"]
    model_path = train_cfg["model_save_path"]
    metrics_out = test_cfg["metrics_out"]

    logging.basicConfig(level=logging.INFO)

    try:
        test_df = pd.read_pickle(test_pkl)

        loader_config = {
            "batch_size": 32,
            "num_workers": 2,
            "transform": None,
        }
        test_loader = create_data_loader(test_df, loader_config)

        device_str = train_cfg["device"]
        device = torch.device(
            device_str if torch.cuda.is_available() else "cpu")
        model = ResNetClassifier(n_classes=10).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        os.makedirs(os.path.dirname(metrics_out), exist_ok=True)

        metrics = test_model(model=model, test_loader=test_loader,
                             device=device, loss_function=nn.CrossEntropyLoss())
        with open(metrics_out, "w") as f:
            import json
            json.dump(metrics, f)

        logging.info(f"Test metrics saved at: {metrics_out}")
        sys.exit(0)
    except Exception as e:
        logging.error(f"Test failed: {e}")
        sys.exit(1)
