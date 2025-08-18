import torch
import torch.nn as nn
import torch.nn.functional as F
import math

################################################################################
# Helper: Compute MMD between two sets of latent samples (z vs. z_prior)
################################################################################
def compute_mmd(z, z_prior, sigma=1.0):
    """
    Compute the MMD (Maximum Mean Discrepancy) between:
        z        ~ q_\phi(z)  (encoder outputs)
        z_prior  ~ p(z)       (samples from N(0, I))
    using an RBF kernel with fixed sigma.
    
    Args:
        z:        (batch_size, n_z)
        z_prior:  (batch_size, n_z)
        sigma:    float, bandwidth of the RBF kernel

    Returns:
        Scalar MMD value.
    """
    # Compute pairwise squared distances
    xx = torch.mm(z, z.t())         # (batch_size, batch_size)
    yy = torch.mm(z_prior, z_prior.t())
    xy = torch.mm(z, z_prior.t())

    x2 = (z * z).sum(dim=1, keepdim=True)
    y2 = (z_prior * z_prior).sum(dim=1, keepdim=True)

    dist_xx = x2 + x2.t() - 2. * xx
    dist_yy = y2 + y2.t() - 2. * yy
    dist_xy = x2 + y2.t() - 2. * xy

    # RBF kernel
    K_xx = torch.exp(-dist_xx / (2.0 * sigma**2))
    K_yy = torch.exp(-dist_yy / (2.0 * sigma**2))
    K_xy = torch.exp(-dist_xy / (2.0 * sigma**2))

    # Remove diagonal elements from K_xx, K_yy for unbiased MMD
    batch_size = z.size(0)
    sum_xx = (K_xx.sum() - K_xx.diagonal().sum()) / (batch_size * (batch_size - 1))
    sum_yy = (K_yy.sum() - K_yy.diagonal().sum()) / (batch_size * (batch_size - 1))
    sum_xy = K_xy.sum() / (batch_size * batch_size)

    mmd = sum_xx + sum_yy - 2.0 * sum_xy
    return mmd


################################################################################
# InfoVAE (MMD-VAE) Class
# - Uses MMD to match q(z|x) to p(z), instead of KL divergence
################################################################################
class InfoVAE(nn.Module):
    def __init__(self, n_x, n_z, n_c, dropout_rate=0.5):
        """
        Args:
          n_x: input dimension (e.g., flattened image size)
          n_z: latent dimension
          n_c: condition dimension (if using conditional info)
          dropout_rate: dropout probability
        """
        super(InfoVAE, self).__init__()
        self.n_x = n_x
        self.n_z = n_z
        self.n_c = n_c

        # ----------------------
        # Encoder layers
        # ----------------------
        self.fc1 = nn.Linear(n_x + n_c, 4096)
        self.fc2 = nn.Linear(4096, 2048)
        self.fc_mu = nn.Linear(2048, n_z)         # mean of q(z|x)
        self.fc_log_sigma = nn.Linear(2048, n_z)  # log-variance of q(z|x)

        self.bn1 = nn.BatchNorm1d(4096)
        self.bn2 = nn.BatchNorm1d(2048)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        # ----------------------
        # Decoder layers
        # ----------------------
        self.fc3 = nn.Linear(n_z + n_c, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.fc5 = nn.Linear(4096, n_x)

        self.bn3 = nn.BatchNorm1d(2048)
        self.bn4 = nn.BatchNorm1d(4096)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.dropout4 = nn.Dropout(dropout_rate)

    def encoder(self, x, c):
        """
        Encode x (and condition c) to produce parameters (mu, log_sigma).
        Returns:
          mu, log_sigma
        """
        # Concatenate input data x and condition c
        x_cat = torch.cat((x, c), dim=1)
        h1 = F.relu(self.bn1(self.fc1(x_cat)))
        h1 = self.dropout1(h1)
        h2 = F.relu(self.bn2(self.fc2(h1)))
        h2 = self.dropout2(h2)

        mu = self.fc_mu(h2)
        log_sigma = self.fc_log_sigma(h2)
        return mu, log_sigma

    def sample_z(self, mu, log_sigma):
        """
        Reparameterization trick: z = mu + exp(log_sigma/2) * epsilon
        where epsilon ~ N(0, I).
        """
        eps = torch.randn_like(mu)
        return mu + torch.exp(log_sigma / 2.0) * eps

    def decoder(self, z, c):
        """
        Decode latent variable z (and condition c) to reconstruct x.
        """
        z_cat = torch.cat((z, c), dim=1)
        h3 = F.relu(self.bn3(self.fc3(z_cat)))
        h3 = self.dropout3(h3)
        h4 = F.relu(self.bn4(self.fc4(h3)))
        h4 = self.dropout4(h4)
        # If your x is in [0,1], using sigmoid is common. Otherwise, you might omit it.
        return torch.sigmoid(self.fc5(h4))

    def simmetrize(self, images):
        """
        (Optional) If you need symmetrical images, as in your original code.
        """
        batch_size = images.size(0)
        image_size = int(math.sqrt(self.n_x))
        images_2d = images.view(batch_size, image_size, image_size)
        symmetric_images = torch.empty_like(images_2d)
        for i in range(batch_size):
            a = images_2d[i]
            symmetric_images[i] = (a.transpose(0, 1) + a) / 2.0
        symmetric_images_flattened = symmetric_images.view(batch_size, -1)
        return symmetric_images_flattened

    def forward(self, x, c):
        """
        Forward pass: encode -> sample -> decode.
        Returns:
          x_recon: reconstructed input
          z: latent sample
          mu, log_sigma: for diagnostics/analysis
        """
        mu, log_sigma = self.encoder(x, c)
        z = self.sample_z(mu, log_sigma)
        x_recon = self.decoder(z, c)
        return x_recon, z, mu, log_sigma

    @staticmethod
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)


################################################################################
# Loss Class (with a new InfoVAE loss function)
################################################################################
class Loss:

    @staticmethod
    def info_vae_loss(x_true, x_recon, z, lambda_mmd=1.0, recon_type='bce'):
        """
        InfoVAE (MMD-VAE) loss = Reconstruction + lambda * MMD(q(z), p(z))
        
        Args:
            x_true:    Ground-truth input
            x_recon:   Reconstructed input
            z:         Latent samples from q(z|x)
            lambda_mmd: Weight for the MMD penalty
            recon_type: 'bce', 'mse', or 'l1' for the reconstruction term
        """
        # 1) Reconstruction loss
        if recon_type == 'bce':
            # For data in [0,1]
            recon_loss = F.binary_cross_entropy(x_recon, x_true, reduction='sum')
        elif recon_type == 'mse':
            recon_loss = F.mse_loss(x_recon, x_true, reduction='sum')
        else:
            # L1
            recon_loss = F.l1_loss(x_recon, x_true, reduction='sum')

        # 2) MMD penalty: samples z ~ q(z|x), and z_prior ~ p(z) = N(0,I)
        z_prior = torch.randn_like(z)
        mmd_penalty = compute_mmd(z, z_prior, sigma=1.0)

        total_loss = recon_loss + lambda_mmd * mmd_penalty
        return total_loss

    @staticmethod
    def perceptual_loss(model_features, y_true_flat, y_pred_flat, y_true_vgg, y_pred_vgg, mu, log_sigma):
        """
        Your existing perceptual loss (which uses BCE + KL + VGG),
        but be aware that in InfoVAE, we typically omit KL or replace it with MMD.
        If you want to keep KL for some reason, you can combine KL + MMD in a hybrid approach.
        """
        y_pred_flat = y_pred_flat.to(y_true_flat.dtype)
        recon = F.binary_cross_entropy(y_pred_flat, y_true_flat, reduction='sum')

        # Standard VAE KL term: remove or comment out if you're purely doing InfoVAE
        kl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())

        # VGG-based feature matching
        y_true_vgg = y_true_vgg.to(y_pred_vgg.dtype)
        features_y_true = model_features(y_true_vgg.float()).detach()  
        features_y_pred = model_features(y_pred_vgg.float()).detach()
        perc_loss = F.mse_loss(features_y_pred, features_y_true)

        total_loss = kl + recon + perc_loss
        return total_loss

    @staticmethod
    def beta_vae_loss(y_true, y_pred, mu, log_sigma, beta):
        """
        Legacy Beta-VAE style: recon + beta * KL
        (If purely doing InfoVAE, do NOT use this in training.)
        """
        recon = F.l1_loss(y_pred, y_true, reduction='sum')
        kl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        return recon + beta * kl

    @staticmethod
    def beta_vae_loss_symmetric(y_true, y_pred, mu, log_sigma, beta):
        """
        Another legacy Beta-VAE style with symmetrical loss.
        """
        recon = F.l1_loss(y_pred, y_true, reduction='sum')
        kl = -0.5 * torch.sum(1 + log_sigma - mu.pow(2) - log_sigma.exp())
        symmetric_loss = F.mse_loss(y_pred, y_pred.transpose(-1, -2))
        return recon + beta*kl + symmetric_loss

    # ------------- Your other mask-based utility losses remain unchanged -------------
    @staticmethod
    def mae_sum(x, y_true, mask, scaling):
        x_masked = x * mask
        y_true_masked = y_true * mask
        error = 0
        for i in range(x_masked.size()[0]):
            x_mask_single = x_masked[i]
            true_single = y_true_masked[i]
            mask_single = mask[i]
            error += (torch.abs(x_mask_single - true_single).sum() / mask_single.sum()) * scaling
        return error

    def mae_arr(x, y_true, mask, scaling):
        x_masked = x * mask
        y_true_masked = y_true * mask
        errors =[]
        for i in range(x_masked.size()[0]):
            x_mask_single = x_masked[i]
            true_single = y_true_masked[i]
            mask_single = mask[i]
            error = (torch.abs(x_mask_single - true_single).sum() / mask_single.sum()) * scaling
            errors.append(error.item())
        return errors

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

    def rmse_arr(x, y_true, mask, scaling):
        x_masked = x * mask
        y_true_masked = y_true * mask
        errors= []
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
            errors.append((rmse * scaling).item())
        return errors

    @staticmethod
    def mae(x, y_true, mask, scaling):
        x_masked = x * mask
        y_true_masked = y_true * mask
        mae = (torch.abs(x_masked - y_true_masked).sum() / mask.sum()) * scaling
        return mae

    def calculate_mean_std(arr_analyze):
        try:
            np_arr = np.concatenate(arr_analyze)
        except:
            np_arr = arr_analyze
        return np.mean(np_arr), np.std(np_arr)




class Equalizer:

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
    def equalize_the_matrix(pred_matrix, number_of_iterations, verbose_number,  eps):
        L = pred_matrix.shape[0]
        for i in range(number_of_iterations):
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
        return a, b, c


class VAE_Utils:

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
        return image, 0.0
