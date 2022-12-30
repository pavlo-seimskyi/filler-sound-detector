import torch
from torch.utils.data import DataLoader

from src.model.data_builder import OneSampleDataset
from src.model.estimator import BaseEstimator


class MultiLayerPerceptron(BaseEstimator):
    def __init__(self, n_features, n_hidden, n_out, dropout_proba=0.2, **kwargs):
        super().__init__(**kwargs)
        self.in_layer = torch.nn.Linear(n_features, n_hidden)
        self.mid_layer = torch.nn.Linear(n_hidden, n_hidden)
        self.out_layer = torch.nn.Linear(n_hidden, n_out)
        self.dropout = torch.nn.Dropout(p=dropout_proba)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        out = self.in_layer(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.mid_layer(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.out_layer(out)
        out = self.sigmoid(out)
        return out.squeeze(dim=1)

    def build_dataloader(self, x, y=None):
        dataset = OneSampleDataset(x, y)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)
