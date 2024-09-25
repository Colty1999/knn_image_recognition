import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from torchvision import transforms, datasets
from autoencoder import MushroomAutoencoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
input_channels = 3
latent_dim = 128
learning_rate = 1e-4
batch_size = 32
num_epochs = 100
model_save_path = "mushroom_autoencoder_2.pth"
input_dir = "reduced_dataset"
output_dir = "reduced_dataset_output"


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder(root=input_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model = MushroomAutoencoder(input_channels=input_channels, latent_dim=latent_dim).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()

model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}")

    for i, (inputs, _) in progress_bar:
        inputs = inputs.to(DEVICE)
        optimizer.zero_grad()

        reconstructed, mu, log_var = model(inputs)

        loss = criterion(reconstructed, inputs)

        kl_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss += kl_divergence

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress_bar.set_postfix(loss=running_loss / (i + 1))

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}")

torch.save(model.state_dict(), model_save_path)


model.eval()
os.makedirs(output_dir, exist_ok=True)
with torch.no_grad():
    for i, (batch, labels) in enumerate(train_loader):
        batch = batch.to(DEVICE)
        recon_batch, _, _ = model(batch)

        recon_batch = recon_batch.view(-1, 3, 256, 256).cpu()

        for j in range(recon_batch.size(0)):
            recon_img = transforms.ToPILImage()(recon_batch[j])
            img_name = f"reconstructed_{i * batch_size + j}.png"
            recon_img.save(os.path.join(output_dir, img_name))
