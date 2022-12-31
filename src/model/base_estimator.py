from typing import *

import numpy as np
import pandas as pd
import torch

from constants import FILLER_LABELING_THRESHOLD
from src import evaluate
from src.evaluate import calculate_metrics


class BaseEstimator(torch.nn.Module):
    def __init__(
        self,
        main_metric: str,
        batch_size: int = 32,
        loss_fn: torch.nn.Module = torch.nn.BCELoss(),
        cutoff_thres: float = FILLER_LABELING_THRESHOLD,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lr_decay_step: int = 5,
        lr_decay_magnitude: float = 0.1,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.main_metric = main_metric
        self.cutoff_thres = cutoff_thres
        self.history = pd.DataFrame()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.lr_decay_step = lr_decay_step
        self.lr_decay_magnitude = lr_decay_magnitude
        self.class_weights = None
        self.scheduler = None
        self.optimizer = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def forward(self, x):
        pass

    def build_dataloader(self, x, y=None):
        pass

    def set_optimizer(self):
        # Reasonable weight decay between 0 and 0.1
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def set_scheduler(self, step_size=5, gamma=0.1):
        # Learning rate decay - every 5 epochs decrease by 10x
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay_magnitude
        )

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, n_epochs=10):
        if self.optimizer is None:
            self.set_optimizer()
            self.set_scheduler()
        self.compute_class_weights(y_train)
        for epoch in range(1, n_epochs + 1):
            y_train_preds = self.train_epoch(x_train, y_train)
            self.store_metrics(epoch, y_train, y_train_preds, name="train")
            num_rows = 1
            if x_valid is not None and y_valid is not None:
                y_valid_preds = self.predict(x_valid)
                self.store_metrics(epoch, y_valid, y_valid_preds, name="valid")
                num_rows += 1
            self.print_last_results(num_rows)

    def train_epoch(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        self.train()
        dataloader = self.build_dataloader(x, y)
        y_pred = torch.Tensor([]).to(self.device)
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            batch_pred = self(inputs)
            self.loss_fn.weight = self.get_class_weights(targets)
            loss = self.loss_fn(batch_pred, targets)
            loss.backward()
            self.optimizer.step()
            y_pred = torch.cat((y_pred, batch_pred.detach()), dim=0)
        self.scheduler.step()  # step after completing the epoch!
        return y_pred

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        dataloader = self.build_dataloader(x)
        y_pred = torch.Tensor([]).to(self.device)
        for inputs in dataloader:
            inputs = inputs.to(self.device)
            with torch.no_grad():
                batch_pred = self(inputs)
            y_pred = torch.cat((y_pred, batch_pred), dim=0)
        return y_pred

    def evaluate(self, x: torch.Tensor, y: torch.Tensor) -> None:
        y_pred = self.predict(x)
        evaluate(y, y_pred, self.cutoff_thres)

    def store_metrics(self, epoch, y_true, y_pred, name):
        y_true, y_pred = y_true.cpu(), y_pred.cpu()
        metrics = calculate_metrics(y_true, y_pred, self.cutoff_thres)
        self.loss_fn.weight = self.get_class_weights(y_true)
        metrics["loss"] = self.loss_fn(y_pred, y_true).item()
        self.add_to_history(epoch=epoch, dataset_name=name, metrics=metrics)

    def compute_class_weights(self, y: torch.Tensor) -> None:
        labels, counts = y.unique(return_counts=True)
        self.class_weights = {
            k: v for k, v in zip(np.flipud(labels), counts / counts.max())
        }

    def get_class_weights(self, y: torch.Tensor) -> torch.Tensor:
        """Define class weight for the loss function."""
        y = y.type(torch.long)
        for target in torch.unique(y.type(torch.long)):
            assert (
                target.item() in self.class_weights.keys()
            ), f"Class {target} not present in the target vector!"
        weights = y.type(torch.double).clone()
        for class_, weight in self.class_weights.items():
            weights[weights == class_] = weight
        return weights

    def add_to_history(
        self, epoch: int, dataset_name: str, metrics: Dict[str, float]
    ) -> None:
        new_history = pd.DataFrame(
            data=[{**{"epoch": epoch, "dataset": dataset_name}, **metrics}]
        )
        self.history = pd.concat((self.history, np.round(new_history, 4)), axis=0)
        self.history.reset_index(drop=True, inplace=True)

    def print_last_results(self, num_rows) -> None:
        selected_rows = self.history.iloc[-num_rows:]
        metrics = selected_rows.melt(
            id_vars=["epoch", "dataset"], value_vars=["loss", self.main_metric]
        )
        epoch = selected_rows["epoch"].iloc[-1]
        text = [f"epoch: {epoch}"] + [
            f"{row['dataset']} {row['variable']}: {row['value']}"
            for _, row in metrics.iterrows()
        ]
        text = " | ".join(text)
        print("=" * len(text))
        print(text)
