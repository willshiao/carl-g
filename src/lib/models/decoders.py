from typing import List
import numpy as np
import torch
from torch import nn
from enum import Enum

from tqdm import tqdm


class DecoderModel(Enum):
    CONCAT_MLP = 'concat_mlp'
    PRODUCT_MLP = 'prod_mlp'


class MlpConcatDecoder(torch.nn.Module):
    """Concatentation-based MLP link predictor."""

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


class MlpProdDecoder(torch.nn.Module):
    """Hadamard-product-based MLP link predictor."""

    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size
        # self.embeddings = embeddings
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        left, right = x[:, : self.embedding_size], x[:, self.embedding_size :]
        return self.net(left * right)

    def predict(self, x):
        return torch.sigmoid(self.forward(x))


class DecoderZoo:
    """Class that allows switching between different link prediction decoders.
    Two models are currently supported:
      - prod_mlp: a Hadamard product MLP
      - mlp: a concatenation-based MLP

    Reads flags from an instance of absl.FlagValues.
    See ../lib/flags.py for flag defaults and descriptions.
    """

    # Note: we use the value of the enums since we read them in as flags
    models = {
        DecoderModel.CONCAT_MLP.value: MlpConcatDecoder,
        DecoderModel.PRODUCT_MLP.value: MlpProdDecoder,
    }

    def __init__(self, flags):
        self.flags = flags

    def init_model(self, model_class, embedding_size):
        flags = self.flags
        if model_class == MlpConcatDecoder:
            if flags.adjust_layer_sizes:
                return MlpConcatDecoder(
                    embedding_size=embedding_size,
                    hidden_size=flags.link_mlp_hidden_size * 2,
                )
            return MlpConcatDecoder(
                embedding_size=embedding_size, hidden_size=flags.link_mlp_hidden_size
            )
        elif model_class == MlpProdDecoder:
            return MlpProdDecoder(
                embedding_size=embedding_size, hidden_size=flags.link_mlp_hidden_size
            )

    @staticmethod
    def filter_models(models: List[str]):
        return [model for model in models if model in DecoderZoo.models]

    def check_model(self, model_name):
        """Checks if a model with the given name exists.
        Raises an error if not.
        """
        if model_name not in self.models:
            raise ValueError(f'Unknown decoder model: "{model_name}"')
        return True

    def get_model(self, model_name, embedding_size):
        """Given a model name, return the corresponding model class."""
        self.check_model(model_name)
        return self.init_model(self.models[model_name], embedding_size)

class GBTLogisticRegression(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        weight_decay: float,
        is_multilabel: bool,
    ):
        super().__init__()

        self.fc = nn.Linear(in_dim, out_dim)

        self._optimizer = torch.optim.AdamW(
            params=self.parameters(),
            lr=0.01,
            weight_decay=weight_decay,
        )

        self._is_multilabel = is_multilabel

        self._loss_fn = (
            nn.BCEWithLogitsLoss()
            if self._is_multilabel
            else nn.CrossEntropyLoss()
        )

        self._num_epochs = 1000
        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        for m in self.modules():
            self.weights_init(m)

        self.to(self._device)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, x):
        return self.fc(x)

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.train()

        X = torch.from_numpy(X).float().to(self._device)
        y = torch.from_numpy(y).to(self._device)

        for _ in tqdm(range(self._num_epochs), desc="Epochs", leave=False):
            self._optimizer.zero_grad()

            pred = self(X)
            loss = self._loss_fn(input=pred, target=y)

            loss.backward()
            self._optimizer.step()

    def predict(self, X: np.ndarray):
        self.eval()

        with torch.no_grad():
            pred = self(torch.from_numpy(X).float().to(self._device))

        if self._is_multilabel:
            return (pred > 0).float().cpu()
        else:
            return pred.argmax(dim=1).cpu()