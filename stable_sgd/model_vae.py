import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
class UnFlatten(nn.Module):
    def forward(self, input, size=512):
        return input.view(input.size(0), size, 1, 1)

class ConvVAE(nn.Module):
    def __init__(self, device, h_dim = 512, z_dim = 512, image_channels=3):
        super(ConvVAE, self).__init__()
        self.device = device
        self.encoder = nn.Sequential(*list(models.resnet18().children())[:-1])

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
                UnFlatten(),
                nn.ConvTranspose2d(h_dim, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            )

        self.classifier = nn.Sequential(nn.Linear(z_dim, 10),)
    def reparameterize(self, mu, logvar):       
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(self.device)
        z = mu + std * esp
        return z
        
    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        # z     = F.normalize(z, dim=-1)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def inference(self, x):
        h = self.encoder(x)
        h = torch.flatten(h, start_dim=1)
        z, mu, logvar = self.bottleneck(h)
        return z

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z, mu, logvar
    
    def classify(self, x):
        return self.classifier(x)

if __name__ == "__main__":
    vae = ConvVAE()
    inp = torch.randn(2,3,32,32)
    out = vae(inp)
    print(out[0].shape)
