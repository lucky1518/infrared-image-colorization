
import torch

from generator import Generator
from discriminator import Discriminator

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

G = Generator().to(device)
D = Discriminator().to(device)

x = torch.randn(1,3,256,256).to(device)
y = torch.randn(1,3,256,256).to(device)

fake = G(x)

print("Generator Output:", fake.shape)

disc = D(x,y)

print("Discriminator Output:", disc.shape)
