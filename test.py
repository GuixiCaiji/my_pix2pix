import config
from gen_model import Generator
from utils import load_checkpoint, test_pics
import torch.optim as optim
from dataset import MyDataset
from torch.utils.data import DataLoader


def test():
    gen = Generator(in_channels=3).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE)
    test_dataset = MyDataset(root_dir="../CycleGAN-and-pix2pix/datasets/init/test")
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    test_pics(gen, test_loader, folder="test")
    

if __name__ == "__main__":
    test()