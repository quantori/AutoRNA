import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import os
import cv2
from torchvision.models import vgg16
import math
import numpy as np
import random

class VAE(nn.Module):

    def __init__(self, n_x, n_z, n_c, dropout_rate=0.5):
        super(VAE, self).__init__()
        self.n_x = n_x
        self.n_z = n_z
        self.n_c = n_c

        self.fc1 = nn.Linear(n_x + n_c, 4096)  # Doubled from 2048 to 4096
        self.fc2 = nn.Linear(4096, 2048)  # Doubled from 1024 to 2048
        self.fc_mu = nn.Linear(2048, n_z)  # Output size stays because it's determined by n_z
        self.fc_log_sigma = nn.Linear(2048, n_z)  # Output size stays because it's determined by n_z

        self.bn1 = nn.BatchNorm1d(4096)  # Adjusted to match fc1 output size
        self.bn2 = nn.BatchNorm1d(2048)  # Adjusted to match fc2 output size

        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)


        self.fc3 = nn.Linear(n_z + n_c, 2048)  # Doubled from 1024 to 2048
        self.fc4 = nn.Linear(2048, 4096)  # Doubled from 2048 to 4096
        self.fc5 = nn.Linear(4096, n_x)  # Output size stays because it's determined by n_x

        self.bn3 = nn.BatchNorm1d(2048)  # Adjusted to match fc3 output size
        self.bn4 = nn.BatchNorm1d(4096)  # Adjusted to match fc4 output size

        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)


    def encoder(self, x, c):
        x = torch.cat((x, c), dim=1)
        h1 = F.relu(self.bn1(self.fc1(x)))
        h1 = self.dropout1(h1)
        h2 = F.relu(self.bn2(self.fc2(h1)))
        h2 = self.dropout2(h2)
        mu = self.fc_mu(h2)
        log_sigma = self.fc_log_sigma(h2)
        return mu, log_sigma

    def sample_z(self, mu, log_sigma):
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_sigma / 2) * eps

    def decoder(self, z, c):
        z = torch.cat((z, c), dim=1)
        h3 = F.relu(self.bn3(self.fc3(z)))
        h3 = self.dropout3(h3)
        h4 = F.relu(self.bn4(self.fc4(h3)))
        h4 = self.dropout4(h4)
        return torch.sigmoid(self.fc5(h4))

    def simmetrize(self, images):
        batch_size = images.size(0)
        image_size = int(math.sqrt(self.n_x))
        images_2d = images.view(batch_size, image_size, image_size)
        symmetric_images = torch.empty_like(images_2d)
        for i in range(batch_size):
            A = images_2d[i]
            symmetric_images[i] = (A.transpose(0, 1) + A) / 2.0
        symmetric_images_flattened = symmetric_images.view(batch_size, -1)
        return symmetric_images_flattened

    def forward(self, x, c):
        mu, log_sigma = self.encoder(x, c)
        z = self.sample_z(mu, log_sigma)
        output = self.decoder(z, c)
        return output, mu, log_sigma

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


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

    """
    @staticmethod
    def simple_vae_loss(y_true, y_pred, mu, log_sigma):
        recon = F.binary_cross_entropy(y_pred, y_true, reduction='sum')
        kl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        return recon + kl
    """
    @staticmethod
    def beta_vae_loss(y_true, y_pred, mu, log_sigma, beta):
        #recon = F.mse_loss(y_pred, y_true, reduction='sum')
        recon = F.l1_loss(y_pred, y_true, reduction='sum')
        kl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        return recon + beta*kl

    @staticmethod
    def mae_sum(x, y_true, mask, scaling):
        x_masked = x * mask
        y_true_masked = y_true * mask
        error=0
        for i in range(x_masked.size()[0]):
            x_mask_single = x_masked[i]
            true_single = y_true_masked[i]
            mask_single = mask[i]
            error += (torch.abs(x_mask_single - true_single).sum() / mask_single.sum()) * scaling
        return error

    @staticmethod
    def rmse_sum(x, y_true, mask, scaling):
        x_masked = x * mask
        y_true_masked = y_true * mask
        error = 0
        for i in range(x_masked.size()[0]):
            x_mask_single = x_masked[i]
            true_single = y_true_masked[i]
            mask_single = mask[i]
            # Calculate squared error
            squared_error = (x_mask_single - true_single) ** 2
            # Calculate mean of squared error over the masked area, then take the square root
            mse = squared_error.sum() / mask_single.sum()
            rmse = torch.sqrt(mse)
            # Accumulate the scaled RMSE
            error += rmse * scaling
        return error
    @staticmethod
    def mae(x, y_true, mask, scaling):
        x_masked =  x * mask
        y_true_masked = y_true * mask
        mae = (torch.abs(x_masked - y_true_masked).sum() / mask.sum()) * scaling
        return mae


class Equalizer():

    @staticmethod
    def equalize(images):
        batch_size = images.size(0)
        image_size = int(math.sqrt(images.size(1)))
        images_2d = images.view(batch_size, image_size, image_size)
        new_images = []
        for i in range(len(images_2d)):
            image = images_2d[i]
            image = Equalizer.equalize_the_matrix(pred_matrix=image,
                                                  number_of_iterations=1000000,
                                                  verbose_number=10000,
                                                  eps=10e-3)
            new_images.append(image)
        new_images = np.array(new_images)
        new_images_tensor = torch.tensor(new_images, dtype=images.dtype, device=images.device)
        images_2d = new_images_tensor.reshape(batch_size, -1)
        return images_2d

    @staticmethod
    def equalize_the_matrix(pred_matrix, number_of_iterations,verbose_number,  eps):
        L = pred_matrix.shape[0]
        pred_last = pred_matrix.clone()
        for i in range(number_of_iterations):
            if i % verbose_number == 0:
                diff = torch.abs(pred_last - pred_matrix).sum()
                pred_last = pred_matrix.clone()
            j1 = random.randint(0, L - 1)
            j2 = random.randint(0, L - 1)
            j3 = random.randint(0, L - 1)
            if (j1 != j2) and (j2 != j3) and (j1 != j3):
                (pred_matrix[j1][j2], pred_matrix[j1][j3], pred_matrix[j2][j3]) = Equalizer.adjust(pred_matrix[j1][j2],
                                                                                                   pred_matrix[j1][j3],
                                                                                                   pred_matrix[j2][j3],
                                                                                                   eps)
        return pred_matrix

    @staticmethod
    def adjust(a, b, c, eps):
        gamma = random.random()
        if a > b + c:
            a = a - gamma * eps * (a - (b + c))
            b = b + gamma * eps / 2.0 * (a - (b + c))
            c = c + gamma * eps / 2.0 * (a - (b + c))
        return (a, b, c)

class VAE_Utils():

    @staticmethod
    def save_images(pred_images, true_images, title_texts, image_size, filesave):
        pred_images = pred_images * 255
        true_images = true_images * 255
        n = len(pred_images)
        fig = plt.figure(figsize=(16, 8))
        for i in range(n):
            # Display the true image
            ax_true = fig.add_subplot(2, n, i + 1)
            ax_true.imshow(true_images[i].reshape(image_size, image_size), cmap='gray')  # Use true_images here
            ax_true.set_title('True: ' + str(title_texts[i]))
            ax_true.axis('off')

            # Display the predicted image
            ax_pred = fig.add_subplot(2, n, i + n + 1)
            ax_pred.imshow(pred_images[i].reshape(image_size, image_size), cmap='gray')  # Use pred_images here
            ax_pred.set_title('Pred: ' + str(title_texts[i]))
            ax_pred.axis('off')
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
