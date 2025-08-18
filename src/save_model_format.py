from vae import VAE
import torch

# Register manually for full-object load
torch.serialization.add_safe_globals({'vae.VAE': VAE})

# Load full model
model = torch.load("experiments/exp1/model/best_model.pth", map_location="cpu")

# Save only the state_dict
torch.save(model.state_dict(), "experiments/exp1/model/best_model_state.pth")

print("✅ Saved model state_dict successfully.")