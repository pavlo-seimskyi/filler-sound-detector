import numpy as np
import pandas as pd
import torch

from constants import FILLER_LABELING_THRESHOLD
from src import evaluate
from src.evaluate import calculate_metrics

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BaseEstimator(torch.nn.Module):
    def __init__(
        self,
        main_metric: str,
        batch_size: int = 32,
        loss_fn=torch.nn.BCELoss(),
        cutoff_thres: float = FILLER_LABELING_THRESHOLD,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.batch_size = batch_size
        self.main_metric = main_metric
        self.cutoff_thres = cutoff_thres
        self.history = pd.DataFrame()
        self.class_weights = None
        self.scheduler = None
        self.optimizer = None

    def forward(self, x):
        pass

    def build_dataloader(self, x, y=None):
        pass

    def set_optimizer(self, learning_rate=3e-4, weight_decay=1e-4):
        # Reasonable weight decay between 0 and 0.1
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

    def set_scheduler(self, step_size=5, gamma=0.1):
        # Learning rate decay - every 5 epochs decrease by 10x
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=step_size, gamma=gamma
        )

    def fit(self, x_train, y_train, x_valid=None, y_valid=None, n_epochs=10):
        if self.optimizer is None:
            self.set_optimizer()
            self.set_scheduler()
        self.compute_class_weights(y_train)
        for epoch in range(1, n_epochs + 1):
            y_train_preds = self.train_epoch(x_train, y_train)
            self.store_metrics(epoch, y_train, y_train_preds, name="train")
            self.print_last_results()
            if x_valid is not None and y_valid is not None:
                y_valid_preds = self.predict(x_valid)
                self.store_metrics(epoch, y_valid, y_valid_preds, name="valid")
                self.print_last_results()

    def store_metrics(self, epoch, y_true, y_pred, name):
        metrics = calculate_metrics(y_true, y_pred, self.cutoff_thres)
        self.loss_fn.weight = self.get_class_weights(y_true)
        metrics["loss"] = self.loss_fn(y_pred, y_true).item()
        self.add_to_history(epoch=epoch, dataset_name=name, metrics=metrics)

    def train_epoch(self, x, y):
        self.train()
        dataloader = self.build_dataloader(x, y)
        y_pred = torch.Tensor([])
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            self.optimizer.zero_grad()
            batch_pred = self(inputs)
            # Define class weight for the loss function
            self.loss_fn.weight = self.get_class_weights(targets)
            loss = self.loss_fn(batch_pred, targets)
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            y_pred = torch.cat((y_pred, batch_pred.cpu().detach()), axis=0)
        return y_pred

    def evaluate(self, x, y):
        y_pred = self.predict(x)
        evaluate(y, y_pred, self.cutoff_thres)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        dataloader = self.build_dataloader(x)
        y_pred = torch.Tensor([])
        for inputs in dataloader:
            inputs = inputs.to(DEVICE)
            with torch.no_grad():
                batch_pred = self(inputs)
            y_pred = torch.cat((y_pred, batch_pred.cpu()), dim=0)
        return y_pred

    def compute_class_weights(self, y):
        if not isinstance(y, np.ndarray):
            y = y.numpy()
        labels, counts = np.unique(y.astype(int), return_counts=True)
        self.class_weights = {
            k: v for k, v in zip(np.flipud(labels), counts / counts.max())
        }

    def get_class_weights(self, y):
        y = y.type(torch.long)
        for target in torch.unique(y.type(torch.long)):
            assert (
                target.item() in self.class_weights.keys()
            ), f"Class {target} not present in the target vector!"
        weights = y.type(torch.double).clone()
        for class_, weight in self.class_weights.items():
            weights[weights == class_] = weight
        return weights

    def add_to_history(self, epoch, dataset_name, metrics):
        new_history = pd.DataFrame(
            data=[{**{"epoch": epoch, "dataset": dataset_name}, **metrics}]
        )
        self.history = pd.concat((self.history, np.round(new_history, 3)), axis=0)
        self.history.reset_index(drop=True)

    def print_last_results(self):
        cols = ["epoch", "dataset", "loss", self.main_metric]
        last_results = self.history.iloc[-1][cols].to_dict()
        text = " | ".join([f"{k}: {v}" for k, v in last_results.items()])
        print("=" * len(text))
        print(text)
