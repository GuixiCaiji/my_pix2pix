import torch
from utils import save_checkpoint, load_checkpoint, save_pics
import torch.nn as nn
import torch.optim as optim
import config
from dataset import MyDataset
from gen_model import Generator
from disc_model import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm

def train(disc, gen, loader, opt_disc, opt_gen, l1_loss, bce):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        # train discriminator
        y_fake = gen(x)
        D_real = disc(x, y)
        D_fake = disc(x, y_fake.detach())
        D_real_loss = bce(D_real, torch.ones_like(D_real))
        D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
        D_loss = (D_real_loss + D_fake_loss) / 2
        
        disc.zero_grad()
        D_loss.backward()
        opt_disc.step()
        
        # train generator
        D_fake = disc(x, y_fake)
        G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
        l1 = l1_loss(y_fake, y) * config.L1_LAMBDA
        G_loss = G_fake_loss + l1

        gen.zero_grad()
        G_loss.backward()
        opt_gen.step()


def main():
    disc = Discriminator(in_channels=3).to(config.DEVICE)
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    bce = nn.BCEWithLogitsLoss()
    L1_loss = nn.L1Loss()

    if config.LOAD_MODEL:
        load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
        load_checkpoint(config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE)
    
    train_dataset = MyDataset(root_dir="../CycleGAN-and-pix2pix/datasets/background_68p/train")
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS)

    val_dataset = MyDataset(root_dir="../CycleGAN-and-pix2pix/datasets/background_68p/val")
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)

    gen.train()
    disc.train()

    for epoch in range(config.NUM_EPOCHS):
        train(disc, gen, train_loader, opt_disc, opt_gen, L1_loss, bce)

        if config.SAVE_MODEL and epoch % 20 == 0:
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC)
        
        if epoch % 5 == 0:
            save_pics(gen, val_loader, epoch, folder="evaluation")



if __name__ == "__main__":
    main()
