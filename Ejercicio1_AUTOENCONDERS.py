import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# -----------------------------
# Parámetros
# -----------------------------
batch_size = 128
num_epochs = 20
latent_dim = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Carga y preprocesado de datos
# -----------------------------
transform = transforms.ToTensor()

train_dataset = datasets.MNIST(
    root="./data", train=True, download=True, transform=transform
)
test_dataset = datasets.MNIST(
    root="./data", train=False, download=True, transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -----------------------------
# Definición del Autoencoder
# -----------------------------
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim)  # sin activación
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 784),
            nn.Sigmoid()  # salida en [0,1]
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec

model = Autoencoder().to(device)

# -----------------------------
# Pérdida y optimizador
# -----------------------------
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# -----------------------------
# Entrenamiento
# -----------------------------
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, _ in train_loader:
        images = images.view(images.size(0), -1).to(device)

        outputs = model(images)
        loss = criterion(outputs, images)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")

# -----------------------------
# Visualización de reconstrucciones
# -----------------------------
model.eval()

digit_examples = {}
reconstructions = {}

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)

        for i in range(len(labels)):
            label = labels[i].item()

            if label not in digit_examples:
                digit_examples[label] = images[i].cpu()

                img = images[i].view(1, -1)
                reconstructed = model(img)
                reconstructions[label] = reconstructed.cpu().view(28, 28)

            if len(digit_examples) == 10:
                break
        if len(digit_examples) == 10:
            break

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(12, 4))

for digit in range(10):
    plt.subplot(2, 10, digit + 1)
    plt.imshow(digit_examples[digit].view(28, 28), cmap="gray")
    plt.axis("off")

    plt.subplot(2, 10, digit + 11)
    plt.imshow(reconstructions[digit], cmap="gray")
    plt.axis("off")

plt.show()
