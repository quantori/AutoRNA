import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
import os
import cv2
from torchvision.models import vgg16
from tqdm import tqdm


class VAE(nn.Module):
    def __init__(self, n_x, n_z, dropout_rate=0.5):
        super(VAE, self).__init__()
        self.n_z = n_z

        self.fc1 = nn.Linear(n_x, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc_mu = nn.Linear(512, n_z)
        self.fc_log_sigma = nn.Linear(512, n_z)

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(n_z, 512)
        self.fc4 = nn.Linear(512, 1024)
        self.fc5 = nn.Linear(1024, n_x)

        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)

    def encoder(self, x):
        h1 = F.relu(self.fc1(x))
        h1 = self.dropout1(h1)
        h2 = F.relu(self.fc2(h1))
        h2 = self.dropout2(h2)
        mu = self.fc_mu(h2)
        log_sigma = self.fc_log_sigma(h2)
        return mu, log_sigma

    def sample_z(self, mu, log_sigma):
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_sigma / 2) * eps

    def decoder(self, z):
        h3 = F.relu(self.fc3(z))
        h3 = self.dropout3(h3)  # Apply dropout after activation
        h4 = F.relu(self.fc4(h3))
        h4 = self.dropout4(h4)  # Apply dropout after activation
        return torch.sigmoid(self.fc5(h4))

    def forward(self, x):
        mu, log_sigma = self.encoder(x)
        z = self.sample_z(mu, log_sigma)
        return self.decoder(z), mu, log_sigma


class Loss():

    @staticmethod
    def perceptual_loss(model_features, y_true_flat, y_pred_flat, y_true_vgg, y_pred_vgg, mu, log_sigma):
        y_pred_flat = y_pred_flat.to(y_true_flat.dtype)
        recon = F.binary_cross_entropy(y_pred_flat, y_true_flat, reduction='sum')
        kl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        assert y_true_vgg.size(1) == 3 and y_pred_vgg.size(1) == 3, "VGG input must have 3 channels"
        y_true_vgg = y_true_vgg.to(y_pred_vgg.dtype)
        features_y_true = model_features(y_true_vgg.float()).detach()  # Detach so we don't backprop through VGG
        features_y_pred = model_features(y_pred_vgg.float()).detach()  # Detach so we don't backprop through VGG
        perc_loss = F.mse_loss(features_y_pred, features_y_true)
        total_loss = kl + recon + perc_loss
        return total_loss

    @staticmethod
    def vae_loss(y_true, y_pred, mu, log_sigma):
        recon = F.binary_cross_entropy(y_pred, y_true, reduction='sum')
        kl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        return recon + kl

class Utils():

    @staticmethod
    def show_images(images, title_texts, image_size, filesave):
        n = len(images)
        fig = plt.figure(figsize=(8, 8))
        for i in range(n):
            ax = fig.add_subplot(1, n, i+1)
            ax.imshow(images[i].reshape(image_size, image_size), cmap='gray')
            ax.set_title(title_texts[i])
            ax.axis('off')
        plt.savefig(filesave)

    @staticmethod
    def get_model_features():
        vgg = vgg16(pretrained=True).features
        if torch.cuda.is_available():
            vgg = vgg.cuda()
        for param in vgg.parameters():
            param.requires_grad = False
        if torch.cuda.is_available():
            vgg = vgg.cuda()
        return vgg


class CustomDataset(Dataset):
    def __init__(self, directory, transform=None):
        """
        Args:
            directory (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.directory = directory
        self.transform = transform
        self.images = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = os.path.join(self.directory, self.images[idx])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if self.transform:
            image = cv2.resize(image, (64, 64))
            image = image / 255.0
        return (image, 0.0)


CONFIG = {
    'batch_size': 64,
    'train_size': 2000,
    'val_size': 200,
    'image_size': 64,
    'n_epoch': 50,
    'n_z' :512,
    'file_generated':"file_generated.jpg",
    'n_images_show' : 16
}


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Adjust mean and std for each channel if necessary
])

model_features = Utils.get_model_features()

custom_dataset = CustomDataset('rna_images_distance', transform=transform)
print(f"Total size of the dataset: {len(custom_dataset)}")

indices = torch.randperm(len(custom_dataset)).tolist()
train_indices = indices[:CONFIG['train_size']]
val_indices = indices[CONFIG['train_size']:CONFIG['train_size'] + CONFIG['val_size']]

train_dataset = Subset(custom_dataset, train_indices)
val_dataset = Subset(custom_dataset, val_indices)

train_loader = DataLoader(dataset=train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=CONFIG['batch_size'], shuffle=False)

def vae_loss(y_true, y_pred, mu, log_sigma):
    recon = F.binary_cross_entropy(y_pred, y_true, reduction='sum')
    kl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
    return kl + recon


n_x = CONFIG['image_size'] * CONFIG['image_size']
n_z = CONFIG['n_z']

model = VAE(n_x, n_z).double()
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(0,CONFIG['n_epoch']):
    model.train()
    train_loss = 0
    train_loader_pbar = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)  # Progress bar for train_loader
    i = 0
    for batch_idx, (x, _) in train_loader_pbar:
        i += 1
        x = x.view(x.size(0), -1)
        y_pred, mu, log_sigma = model(x)
        loss = vae_loss(x, y_pred, mu, log_sigma)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_loss_vis = train_loss / CONFIG['batch_size'] / i

    train_loss /= len(train_loader.dataset)
    print(f'Epoch {epoch}, Training Loss: {train_loss_vis:.8f}')
    # Optionally, you can still print the loss if you want to keep a log
    # print(f'Epoch {epoch}, Training Loss: {train_loss_vis:.8f}')

n_images = CONFIG['n_images_show']
with torch.no_grad():
    sample_z = torch.randn(n_images, n_z).double()
    generated_images = model.decoder(sample_z).numpy()

Utils.show_images(generated_images, np.arange(0, n_images, 1), CONFIG['image_size'], CONFIG['file_generated'])