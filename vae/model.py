import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, input_channels, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, hidden_dim // 8, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim // 8, hidden_dim // 4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_dim // 4, hidden_dim // 2, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=4, stride=2, padding=1)

        self.fc1 = nn.Linear(hidden_dim * 8 * 8, latent_dim)
        self.fc2 = nn.Linear(hidden_dim * 8 * 8, latent_dim)
        
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.dropout2d(x, 0.5)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.dropout2d(x, 0.5)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.dropout2d(x, 0.5)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = F.dropout2d(x, 0.5)
        
        x = x.view(x.size(0), -1)
        mean = self.fc1(x)
        log_variance = self.fc2(x)

        return mean, log_variance

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_channels):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, hidden_dim * 4 * 4)

        self.deconv1 = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_dim // 2, hidden_dim // 4, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(hidden_dim // 4, hidden_dim // 8, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(hidden_dim // 8, output_channels, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(output_channels, output_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        
        x = F.leaky_relu(self.deconv1(x), 0.2)
        x = F.dropout2d(x, 0.5)
        x = F.leaky_relu(self.deconv2(x), 0.2)
        x = F.dropout2d(x, 0.5)
        x = F.leaky_relu(self.deconv3(x), 0.2)
        x = F.dropout2d(x, 0.5)
        x = F.leaky_relu(self.deconv4(x), 0.2)
        x = F.dropout2d(x, 0.5)
        x = torch.sigmoid(self.deconv5(x))
        
        return x

class Model(nn.Module):
    def __init__(self, Encoder, Decoder, device):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.device = device

    def reparameterization(self, mean, variance):
        # Separate out the randomness into the epsilon term
        epsilon = torch.randn_like(variance, self.device)

        # Now gradients can flow back through mean and variance stil
        z = mean + variance * epsilon

        return z

    def forward(self, x):
        mean, log_variance = self.Encoder(x)

        # Use the reparameterization trick to keep randomness differentiable
        z = self.reparameterization(mean, torch.exp(0.5 * log_variance))

        x_hat = self.Decoder(z)
        return x_hat, mean, log_variance


def bce_loss(x, x_hat, mean, log_variance):
    # reconstruction loss encourages latents to model distribution better
    reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')

    # kl div penalizes latents from deviating too far from gaussian
    kl_divergence = - 0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp())

    # both balance each other out to make a good approximation
    return reconstruction_loss + kl_divergence
