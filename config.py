import torch
import torchvision.transforms as transforms

DEVICE = "cuda:7" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 2e-4
BATCH_SIZE = 8
NUM_WORKERS = 2
IMAGE_SIZE = 512
CHANNELS_IMG = 3
L1_LAMBDA = 100
NUM_EPOCHS = 500
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "disc.pth.tar"
CHECKPOINT_GEN = "gen.pth.tar"

transform_both = transforms.Compose(
    [transforms.Resize(512, 512),
     #  transforms.RandomHorizontalFlip(p=0.1),
     ]
)

transform_input = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]
)
