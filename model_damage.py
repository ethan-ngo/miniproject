import torch
import torch.nn as nn

class LatentVariableClassifier(nn.Module):
    def __init__(self, z_dim=256, h_dim=64, num_classes=11):
        super(LatentVariableClassifier, self).__init__()

		# Map z → latent distribution parameters (μ_h, logvar_h)
        self.fc_mu = nn.Linear(z_dim, h_dim)
        self.fc_logvar = nn.Linear(z_dim, h_dim)

        # Map latent h → labels
        self.fc_out = nn.Linear(h_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def forward(self, z):
        # Infer latent distribution
        mu_h = self.fc_mu(z)
        logvar_h = self.fc_logvar(z)

        # Sample latent variable
        h = self.reparameterize(mu_h, logvar_h)

        # Predict labels
        logits = self.fc_out(h)

        return logits, mu_h, logvar_h

    def compute_loss(self, logits, labels, mu_h, logvar_h, beta=1.0):
        # Multi-label classification loss
        bce = F.binary_cross_entropy_with_logits(logits, labels)

        # KL divergence loss
        kl = -0.5 * torch.mean(
            1 + logvar_h - mu_h.pow(2) - logvar_h.exp()
        )

        total_loss = bce + beta * kl

        return total_loss, bce, kl