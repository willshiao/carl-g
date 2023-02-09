import torch
import torch.nn as nn
from torch_geometric.nn import BatchNorm, GCNConv, LayerNorm, Sequential, SAGEConv, GATConv
from torch_geometric.data import Data
import torch.nn.functional as F
from enum import Enum


class EncoderModel(Enum):
    GCN = 'gcn'


class GCN(nn.Module):
    """Basic GCN encoder.
    This is based off of the official BGRL encoder implementation.
    """

    def __init__(
        self,
        layer_sizes,
        batchnorm=False,
        batchnorm_mm=0.99,
        layernorm=False,
        weight_standardization=False,
        use_feat=True,
        n_nodes=0,
        batched=False,
    ):
        super().__init__()

        assert batchnorm != layernorm
        assert len(layer_sizes) >= 2
        self.n_layers = len(layer_sizes)
        self.batched = batched
        self.input_size, self.representation_size = layer_sizes[0], layer_sizes[-1]
        self.weight_standardization = weight_standardization

        layers = []
        relus = []
        batchnorms = []

        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            if batched:
                layers.append(GCNConv(in_dim, out_dim))
                relus.append(nn.PReLU())
                if batchnorm:
                    batchnorms.append(BatchNorm(out_dim, momentum=batchnorm_mm))
            else:
                layers.append(
                    (GCNConv(in_dim, out_dim), 'x, edge_index -> x'),
                )

                if batchnorm:
                    layers.append(BatchNorm(out_dim, momentum=batchnorm_mm))
                else:
                    layers.append(LayerNorm(out_dim))

                layers.append(nn.PReLU())

        if batched:
            self.convs = nn.ModuleList(layers)
            self.relus = nn.ModuleList(relus)
            self.batchnorms = nn.ModuleList(batchnorms)
        else:
            self.model = Sequential('x, edge_index', layers)

        self.use_feat = use_feat
        if not self.use_feat:
            self.node_feats = nn.Embedding(n_nodes, layer_sizes[1])

    def split_forward(self, x, edge_index):
        """Convenience function to perform a forward pass on a feature matrix
        and edge index separately without needing to create a Data object.
        """
        return self(Data(x, edge_index))

    def forward(self, data):
        if not self.batched:
            if self.weight_standardization:
                self.standardize_weights()
            if self.use_feat:
                return self.model(data.x, data.edge_index)
            return self.model(self.node_feats.weight.data.clone(), data.edge_index)
        # otherwise, batched
        x = data.x
        for i, conv in enumerate(self.convs):
            x = conv(x, data.edge_index)
            x = self.relus[i](x)
            x = self.batchnorms[i](x)
        return x

    def reset_parameters(self):
        self.model.reset_parameters()

    def standardize_weights(self):
        skipped_first_conv = False
        for m in self.model.modules():
            if isinstance(m, GCNConv):
                if not skipped_first_conv:
                    skipped_first_conv = True
                    continue
                weight = m.lin.weight.data
                var, mean = torch.var_mean(weight, dim=1, keepdim=True)
                weight = (weight - mean) / (torch.sqrt(var + 1e-5))
                m.lin.weight.data = weight

    def get_node_feats(self):
        if hasattr(self, 'node_feats'):
            return self.node_feats
        return None

    @property
    def num_layers(self):
        return self.n_layers

class GraphSAGE_GCN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size):
        super().__init__()

        self.convs = nn.ModuleList([
            SAGEConv(input_size, hidden_size, root_weight=True),
            SAGEConv(hidden_size, hidden_size, root_weight=True),
            SAGEConv(hidden_size, embedding_size, root_weight=True),
        ])

        self.skip_lins = nn.ModuleList([
            nn.Linear(input_size, hidden_size, bias=False),
            nn.Linear(input_size, hidden_size, bias=False),
            ])

        self.layer_norms = nn.ModuleList([
            LayerNorm(hidden_size),
            LayerNorm(hidden_size),
            LayerNorm(embedding_size),
        ])

        self.activations = nn.ModuleList([
            nn.PReLU(1),
            nn.PReLU(1),
            nn.PReLU(1),
        ])

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        batch = data.batch if hasattr(data, 'batch') else None

        h1 = self.convs[0](x, edge_index)
        h1 = self.layer_norms[0](h1, batch)
        h1 = self.activations[0](h1)

        x_skip_1 = self.skip_lins[0](x)
        h2 = self.convs[1](h1 + x_skip_1, edge_index)
        h2 = self.layer_norms[1](h2, batch)
        h2 = self.activations[1](h2)

        x_skip_2 = self.skip_lins[1](x)
        ret = self.convs[2](h1 + h2 + x_skip_2, edge_index)
        ret = self.layer_norms[2](ret, batch)
        ret = self.activations[2](ret)
        return ret

    def reset_parameters(self):
        for m in self.convs:
            m.reset_parameters()
        for m in self.skip_lins:
            m.reset_parameters()
        for m in self.activations:
            m.weight.data.fill_(0.25)
        for m in self.layer_norms:
            m.reset_parameters()


class EncoderZoo:
    """Returns an encoder of the specified type.
    Reads flags from an instance of absl.FlagValues.
    See ../lib/flags.py for flag defaults and descriptions.
    """

    # Note: we use the value of the enums since we read them in as flags
    models = {EncoderModel.GCN.value: GCN}

    def __init__(self, flags):
        self.flags = flags

    def _init_model(
        self,
        model_class,
        input_size: int,
        use_feat: bool,
        n_nodes: int,
        batched: bool,
        n_feats: int,
    ):
        flags = self.flags
        if model_class == GCN:
            return GCN(
                [input_size] + flags.graph_encoder_layer_dims,
                batchnorm=True,
                use_feat=use_feat,
                n_nodes=n_nodes,
                batched=batched,
            )

    @staticmethod
    def check_model(model_name: str):
        """Checks if a model with the given name exists.
        Raises an error if not.
        """
        if model_name not in EncoderZoo.models:
            raise ValueError(f'Unknown encoder model: "{model_name}"')
        return True

    def get_model(
        self,
        model_name: str,
        input_size: int,
        use_feat: bool,
        n_nodes: int,
        n_feats: int,
        batched: bool = False,
    ):
        EncoderZoo.check_model(model_name)
        return self._init_model(
            EncoderZoo.models[model_name],
            input_size,
            use_feat,
            n_nodes,
            batched=batched,
            n_feats=n_feats,
        )

class GATEncoder(nn.Module):

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()

        self._conv1 = GATConv(in_dim, 256, heads=4, concat=True)
        self._skip1 = nn.Linear(in_dim, 4 * 256)

        self._conv2 = GATConv(4 * 256, 256, heads=4, concat=True)
        self._skip2 = nn.Linear(4 * 256, 4 * 256)

        self._conv3 = GATConv(4 * 256, out_dim, heads=6, concat=False)
        self._skip3 = nn.Linear(4 * 256, out_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.elu(self._conv1(x, edge_index) + self._skip1(x))
        x = F.elu(self._conv2(x, edge_index) + self._skip2(x))
        x = self._conv3(x, edge_index) + self._skip3(x)

        return x
