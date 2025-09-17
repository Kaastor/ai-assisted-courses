import torch

from lab6_generative.train_vae import VAE


def test_vae_forward_shapes():
    model = VAE(latent_dim=8)
    x = torch.rand(4, 1, 28, 28)
    recon, loss, recon_loss, kl_loss = model(x, beta=1.0)
    assert recon.shape == (4, 1, 28, 28)
    assert isinstance(loss, torch.Tensor)
    assert recon_loss > 0
    assert kl_loss > 0
