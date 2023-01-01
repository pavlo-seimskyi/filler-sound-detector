from torch.utils.data import Dataset


class OneSampleDataset(Dataset):
    def __init__(self, x, y=None):
        self.x = x
        self.y = y
        if y is not None:
            assert x.size(0) == y.size(0), "X and y must have equal shape."

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, index):
        if self.y is not None:
            return self.x[index], self.y[index]
        else:
            return self.x[index]
