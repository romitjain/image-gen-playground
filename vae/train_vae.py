import torch
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from matplotlib import pyplot as plt

from src.utils import find_files, num_params, show_comparison, show_image
from vae.dataset import LogoDataset
from vae.model import Encoder, Decoder, Model, bce_loss

if __name__ == '__main__':
    """
    Train a Variational Autoencoder on the Logo dataset
    python -m vae.train_vae \
        --dataset_path ./dataset/ \
        --img_size 128 --batch_size 256 --hidden_dim 1024 --latent_dim 256 \
        --lr 3e-4 --epochs 100
    """
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='./dataset/')
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=1024)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=100)

    args = parser.parse_args()
    img_size = (args.img_size, args.img_size)
    batch_size = args.batch_size
    hidden_dim = args.hidden_dim
    latent_dim = args.latent_dim
    lr = args.lr
    epochs = args.epochs
    dataset_path = args.dataset_path

    imgs = find_files(dataset_path, ext=('.jpg', '.jpeg', '.png'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = LogoDataset(imgs[:-200], img_size)
    test_ds = LogoDataset(imgs[-200:], img_size)

    print(len(train_ds), len(test_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=2
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        prefetch_factor=2
    )

    encoder = Encoder(input_channels=3, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim=hidden_dim, output_channels = 3)

    model = Model(Encoder=encoder, Decoder=decoder, device=device).to(device) # Bad pattern

    print(f'Number of parameters: {num_params(model)}')

    if device == torch.device("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
        torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        torch.set_float32_matmul_precision("high")

        model = torch.compile(model)
        # warmup the model
        input = torch.randn((1, 3, 128, 128), device=device)
        for _ in range(10):
            model(input)

    optimizer = Adam(model.parameters(), lr=lr)

    ### Training the model

    model.train()

    all_losses = []
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in tqdm(enumerate(train_loader), total=len(train_loader)):
            x = x.to(device)

            optimizer.zero_grad()

            x_hat, mean, log_variance = model(x)
            loss = bce_loss(x, x_hat, mean, log_variance)

            overall_loss += loss.item()

            loss.backward()
            optimizer.step()
            all_losses.append(loss.item())

        print(f"epoch {epoch + 1}: average loss {overall_loss / (batch_idx*batch_size)}")

    plt.figure()
    plt.title("Loss")
    plt.plot(all_losses)

    ### Sampling from the model

    model.eval()

    x = next(iter(test_loader))
    with torch.no_grad():
        x = x.to(device)
        x_hat, _, _ = model(x)

    show_comparison(x, x_hat, 1)
    show_comparison(x, x_hat, 4)

    save_image(x, 'original.png')
    save_image(x_hat, 'reconstruction.png')

    ### Generate new images
    with torch.no_grad():
        noise = torch.randn((batch_size, latent_dim), device=device)
        generated_images = decoder(noise)

    show_image(generated_images, img_size=img_size, idx = 6)
    show_image(generated_images, img_size=img_size, idx = 7)

    model.save('./models/vae/model.pth')
