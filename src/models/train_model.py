# Intent: Using independent and dependent variable as input, train model.
import torch
import torch.nn as nn
from torch.distributions import Categorical, Normal, MixtureSameFamily
from torch.optim.swa_utils import AveragedModel, SWALR


class MDN(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_components):
        """
        Mixture density network.
        in_dim: Number of input variables
        out_dim: Number of output variables
        hidden_dim: Size of hidden layer
        n_components: Number of normal distributions to use in mixture
        """
        super().__init__()
        self.n_components = n_components
        # Last layer output dimension rationale:
        # Need two parameters for each distributionm thus 2 * n_components.
        # Need each of those for each output dimension, thus that multiplication
        self.norm_network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, 2 * n_components * out_dim),
        )
        self.cat_network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(),
            nn.Linear(hidden_dim, n_components * out_dim),
        )

    def forward(self, x):
        norm_params = self.norm_network(x)
        # Split so we get parameters for mean and standard deviation
        mean, std = torch.split(norm_params, norm_params.shape[1] // 2, dim=1)
        # We need rightmost dimension to be n_components for mixture
        mean = mean.view(mean.shape[0], -1, self.n_components)
        std = std.view(std.shape[0], -1, self.n_components)
        normal = Normal(mean, torch.exp(std))

        cat_params = self.cat_network(x)
        # Again, rightmost dimension must be n_components
        cat = Categorical(
            logits=cat_params.view(cat_params.shape[0], -1, self.n_components)
        )

        return MixtureSameFamily(cat, normal)


def train_model(indep_vars, dep_var, verbose=True):
    """
    Trains MDNVol network. Uses AdamW optimizer with cosine annealing learning rate schedule.
    Ouputs averaged model over the last 25% of training epochs.

    indep_vars: n x m torch tensor containing independent variables
        n = number of data points
        m = number of input variables
    dep_var: n x 1 torch tensor containing single dependent variable
        n = number of data points
        1 = single output variable
    """
    model = MDN(indep_vars.shape[1], 1, 250, 5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100, 2)

    swa_model = AveragedModel(model)
    swa_start = 750
    swa_scheduler = SWALR(
        optimizer, swa_lr=0.001, anneal_epochs=10, anneal_strategy="cos"
    )

    model.train()
    swa_model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        output = model(indep_vars)

        loss = -output.log_prob(dep_var).sum()

        loss.backward()
        optimizer.step()

        if epoch > swa_start:
            swa_model.update_parameters(model)
            swa_scheduler.step()
        else:
            scheduler.step()

        if epoch % 10 == 0:
            if verbose:
                print(f"Epoch {epoch} complete. Loss: {loss}")

    swa_model.eval()
    return swa_model
