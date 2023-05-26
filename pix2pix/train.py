from pathlib import Path

import easydict
import torch
import yaml
from easydict import EasyDict
from loguru import logger
from torch import nn
from tqdm import tqdm

from pix2pix.utils import save_checkpoint, save_some_examples, load_checkpoint
from qartezator.data.datamodule import QartezatorDataModule
from pix2pix.discriminators import Discriminator
from pix2pix.generators import UnetGenerator
from qartezator.train import get_args
from torch.utils.tensorboard import SummaryWriter
torch.backends.cudnn.benchmark = True


def train_step(generator: nn.Module, discriminator: nn.Module, data_loader: torch.utils.data.DataLoader,
               gen_optimizer: torch.optim.Optimizer, disc_optimizer: torch.optim.Optimizer,
               loss_fn: torch.nn.Module, bce_loss: torch.nn.Module, config: easydict.EasyDict, device='cuda'):
    gen_loss = []
    dis_loss = []
    loop = tqdm(data_loader)
    for idx, (X, y) in enumerate(loop):
        X, y = X.to(device), y.to(device)
        # Train Discriminator
        y_fake = generator(X)
        d_real = discriminator(X, y)
        d_real_loss = bce_loss(d_real, torch.ones_like(d_real))
        d_fake = discriminator(X, y_fake.detach())
        d_fake_loss = bce_loss(d_fake, torch.zeros_like(d_fake))
        d_loss = (d_real_loss + d_fake_loss) / 2

        discriminator.zero_grad()
        dis_loss.append(d_loss.item())
        d_loss.backward()
        disc_optimizer.step()

        # Train Generator
        d_fake = discriminator(X, y_fake)
        g_fake_loss = bce_loss(d_fake, torch.ones_like(d_fake))
        loss = loss_fn(y_fake, y) * config.training_model.l1_lambda
        g_loss = g_fake_loss + loss

        gen_optimizer.zero_grad()
        gen_loss.append(g_loss.item())
        g_loss.backward()
        gen_optimizer.step()

        logger.info(f'Gen loss: {g_loss}, Dis loss: {d_loss}')
        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(d_real).mean().item(),
                D_fake=torch.sigmoid(d_fake).mean().item(),
            )
    return gen_loss, dis_loss


def main():
    args = get_args()

    # read config
    with open(args.config_path, "r") as f:
        try:
            config = EasyDict(yaml.safe_load(f))
        except yaml.YAMLError as exc:
            print(exc)
    root_path = Path(__file__).resolve().parents[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator_loss = []
    discriminator_loss = []
    discriminator = Discriminator(in_channels=3).cuda()
    generator = UnetGenerator(in_channels=3).cuda()
    discriminator_opt = torch.optim.Adam(discriminator.parameters(), lr=config.training_model.lr,
                                         betas=(config.training_model.beta, 0.999))
    generator_opt = torch.optim.Adam(generator.parameters(), lr=config.training_model.lr,
                                     betas=(config.training_model.beta, 0.999))
    bce_loss = nn.BCEWithLogitsLoss()
    l1_loss = nn.L1Loss()

    if config.training_model.load_model:
        load_checkpoint(
            root_path.joinpath(config.training_model.checkpoint_gen), generator,
            generator_opt, config.training_model.lr, device=device
        )
        load_checkpoint(
        root_path.joinpath(config.training_model.checkpoint_disc), discriminator,
            discriminator_opt, config.training_model.lr, device=device
        )

    dm = QartezatorDataModule(
        root_path=root_path.joinpath(config.location.data_root_dir),
        train_txt_path=root_path.joinpath(config.datamodule.train_txt_path),
        val_txt_path=root_path.joinpath(config.datamodule.val_txt_path),
        test_txt_path=root_path.joinpath(config.datamodule.test_txt_path),
        input_size=config.datamodule.input_size
    )
    train_dataloader = dm.train_dataloader()
    val_dataloader = dm.val_dataloader()

    for epoch in range(config.training_model.max_epochs):
        logger.info(f'Training epoch {epoch}')
        gen_loss, dis_loss = train_step(
            generator, discriminator, train_dataloader, generator_opt,
            discriminator_opt, l1_loss, bce_loss, config=config, device=device
        )
        generator_loss.extend(gen_loss)
        discriminator_loss.extend(dis_loss)
        if config.training_model.save_model and epoch % 10 == 0:
            save_checkpoint(
                generator, generator_opt,
                filename=root_path.joinpath(config.training_model.model_dir).joinpath(f'generator_{epoch}.pth'))
            save_checkpoint(
                discriminator, discriminator_opt,
                filename=root_path.joinpath(config.training_model.model_dir).joinpath(f'discriminator_{epoch}.pth'))
        if epoch % 10 == 0:
            save_some_examples(generator, val_dataloader, epoch, folder=config.location.out_root_dir, device=device)


if __name__ == '__main__':
    main()
