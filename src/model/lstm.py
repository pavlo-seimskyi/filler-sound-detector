import torch
from torch.utils.data import DataLoader

from src.model.base_estimator import BaseEstimator
from src.model.data_builder import OneSampleDataset


class LSTM(BaseEstimator):
    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_layers: int,
        n_out: int,
        dropout_proba=0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.lstm = torch.nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_proba,
        )
        self.mid_layer = torch.nn.Linear(n_hidden, n_hidden)
        self.out_layer = torch.nn.Linear(n_hidden, n_out)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # hidden & internal state
        h_0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).to(self.device)
        c_0 = torch.zeros(self.n_layers, x.size(0), self.n_hidden).to(self.device)

        _, (h_n, c_n) = self.lstm(x, (h_0, c_0))
        # n_layers, batch_size, hidden_size
        h_n = h_n.view(self.n_layers, x.size(0), self.n_hidden)
        # take last layer's hidden state
        h_n = h_n[-1]

        out = self.relu(h_n)
        out = self.mid_layer(out)
        out = self.relu(out)
        out = self.out_layer(out)
        return self.sigmoid(out)

    def build_dataloader(self, x, y=None):
        # We will create non-overlapping sequences for now
        # x = x.unsqueeze(dim=1)
        # y = y.unsqueeze(dim=1) if y is not None else None
        dataset = OneSampleDataset(x, y)
        return DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=False)
