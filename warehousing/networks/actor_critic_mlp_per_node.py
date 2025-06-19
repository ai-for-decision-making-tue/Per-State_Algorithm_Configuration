import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, BatchNorm1d


class NodeMLPActor4Time(torch.nn.Module):
    def __init__(self, input_dim, output_dims, hidden_dim, n_nodes, min_val=torch.finfo(torch.float).min):
        super(NodeMLPActor4Time, self).__init__()

        self.n_nodes = n_nodes
        self.num_features = input_dim
        self.min_val = min_val

        # self.lin1 = Linear(self.num_features, hidden_dim * 2)
        self.lin1 = Linear(self.num_features, hidden_dim)
        # self.batch_norm = BatchNorm1d(2 * hidden_dim)
        self.batch_norm = BatchNorm1d(hidden_dim)
        self.relu = ReLU()
        self.lin2 = Linear(hidden_dim * 2, hidden_dim)
        self.softmax = torch.nn.Softmax(dim=1)

        self.lin_dist = Linear(4, hidden_dim)
        self.lin_time = Linear(1, hidden_dim)

        self.out_layers = torch.nn.ModuleList()
        for i, output_dim in enumerate(output_dims):
            self.out_layers.append(Linear(3 * hidden_dim, output_dim))

        self.softmax = torch.nn.Softmax(dim=1)

    # Dict input
    def forward(self, observations, state, info={}):

        # Reshape to infer batch_size
        x = observations['obs'].reshape(-1, 5 + self.n_nodes * self.num_features)
        batch_size = x.shape[0]
        time_until_change = x[:, 0].reshape(-1, 1)
        dist_type = x[:, 1:5].reshape(-1, 4)
        x = x[:, 5:].reshape(-1, self.num_features)

        x = x.reshape(-1, self.num_features)

        x = self.lin1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.reshape(batch_size, self.n_nodes, -1)

        x = x.mean(dim=1)

        dist_type = self.lin_dist(dist_type)
        dist_type = self.relu(dist_type)

        time_until_change = self.lin_time(time_until_change)
        time_until_change = self.relu(time_until_change)

        x = torch.cat([x, dist_type, time_until_change], dim=1)

        outs = []
        for out_layer in self.out_layers:
            outs.append(self.softmax(out_layer(x)).unsqueeze(dim=1))

        x = torch.cat(outs, dim=1)

        return x, state


class NodeMLPCritic4Time(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, n_nodes):
        super(NodeMLPCritic4Time, self).__init__()

        self.n_nodes = n_nodes
        self.num_features = input_dim

        # self.lin1 = Linear(self.num_features, 2 * hidden_dim)
        self.lin1 = Linear(self.num_features, hidden_dim)
        self.batch_norm = BatchNorm1d(hidden_dim)
        self.relu = ReLU()
        self.lin2 = Linear(2 * hidden_dim, hidden_dim)

        self.lin_dist = Linear(4, hidden_dim)
        self.lin_time = Linear(1, hidden_dim)

        self.lin3 = Linear(3 * hidden_dim, 1)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:

        x = observations['obs'].reshape(-1, 5 + self.n_nodes * self.num_features)
        batch_size = x.shape[0]
        time_until_change = x[:, 0].reshape(-1, 1)
        dist_type = x[:, 1:5].reshape(-1, 4)
        x = x[:, 5:].reshape(-1, self.num_features)

        x = self.lin1(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = x.reshape(batch_size, self.n_nodes, -1)
        x = x.mean(dim=1)

        dist_type = self.lin_dist(dist_type)
        dist_type = self.relu(dist_type)

        time_until_change = self.lin_time(time_until_change)
        time_until_change = self.relu(time_until_change)

        x = torch.cat([x, dist_type, time_until_change], dim=1)

        x = self.lin3(x)

        return x
