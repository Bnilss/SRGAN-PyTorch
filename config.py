# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# ==============================================================================
# File description: Realize the parameter configuration function of data set, model, training and verification code.
# ==============================================================================
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from model import ContentLoss, Discriminator, Generator
from argparse import ArgumentParser


parser = ArgumentParser("SRGAN", description="Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network.")
parser.add_argument("-mode", help="train or valid", default="valid")
parser.add_argument("-seed", default=42, type=int)
parser.add_argument("-no-cudnn-bench", action="store_false")
parser.add_argument("-upscale-factor", default=4, type=int)
parser.add_argument("-train-dir", help="path to the training dataset", default="data/ImageNet/train")
parser.add_argument("-valid-dir", help="path to the validation dataset", default="data/ImageNet/valid")
parser.add_argument("-weights")
parser.add_argument("-device", default="cuda:0" if torch.cuda.is_available() else "cpu")
parser.add_argument("-exp-name", default="exp000")
parser.add_argument("-bs", type=int, default=16)
parser.add_argument("-epochs", type=int, default=10)
parser.add_argument("-img-size", default=96, type=int)
parser.add_argument("-test-paths", nargs='+', default=(None, None))
parser.add_argument("-save-dir", default="results")

args = parser.parse_args()

# ==============================================================================
#                              Common configure
# ==============================================================================
torch.manual_seed(args.seed)               # Set random seed.
upscale_factor   = args.upscale_factor     # How many times the size of the high-resolution image in the data set is than the low-resolution image.
device           = torch.device(args.device)  # Use the first GPU for processing by default.
cudnn.benchmark  = args.no_cudnn_bench     # If the dimension or type of the input data of the network does not change much, turn it on, otherwise turn it off.
mode             = args.mode                # Run mode. Specific mode loads specific variables.
exp_name         = args.exp_name                # Experiment name.

# ==============================================================================
#                              Train configure
# ==============================================================================
if mode == "train":
    # Configure dataset.
    train_dir             = args.train_dir       # The address of the training dataset.
    valid_dir             = args.valid_dir      # The address of the validating dataset.
    image_size            = args.img_size                          # High-resolution image size in the training dataset.
    batch_size            = args.bs                          # Dataset batch size.

    # Configure model.
    discriminator         = Discriminator().to(device)  # Load the discriminator model.
    generator             = Generator().to(device)      # Load the generator model.

    # Resume training.
    start_p_epoch         = 0                           # The number of initial iterations of the generator training phase. When set to 0, it means incremental training.
    start_epoch           = 0                           # The number of initial iterations of the adversarial training phase. When set to 0, it means incremental training.
    resume                = False                       # Set to `True` to continue training from the previous training progress.
    resume_p_weight       = ""                          # Restore the weight of the generator model during generator training.
    resume_d_weight       = ""                          # Restore the weight of the generator model during the training of the adversarial network.
    resume_g_weight       = ""                          # Restore the weight of the discriminator model during the training of the adversarial network.

    # Train epochs.
    p_epochs              = 46                          # The total number of epochs of the generator training phase.
    epochs                = args.epochs                          # The total number of epochs of the adversarial training phase.

    # Loss function.
    psnr_criterion        = nn.MSELoss().to(device)     # PSNR metrics.
    pixel_criterion       = nn.MSELoss().to(device)     # Pixel loss.
    content_criterion     = ContentLoss().to(device)    # Content loss.
    adversarial_criterion = nn.BCELoss().to(device)     # Adversarial loss.
    # Perceptual loss function weight.
    pixel_weight          = 0.01
    content_weight        = 1.0
    adversarial_weight    = 0.001

    # Optimizer.
    p_optimizer           = optim.Adam(generator.parameters(),     0.0001, (0.9, 0.999))  # Generator model learning rate during generator network training.
    d_optimizer           = optim.Adam(discriminator.parameters(), 0.0001, (0.9, 0.999))  # Discriminator learning rate during adversarial network training.
    g_optimizer           = optim.Adam(generator.parameters(),     0.0001, (0.9, 0.999))  # Generator learning rate during adversarial network training.

    # Scheduler.
    d_scheduler           = StepLR(d_optimizer, epochs // 2, 0.1)  # Discriminator model scheduler during adversarial network training.
    g_scheduler           = StepLR(g_optimizer, epochs // 2, 0.1)  # Generator model scheduler during adversarial network training.

    # Training log.
    writer                = SummaryWriter(os.path.join("samples",  "logs", exp_name))

    # Additional variables.
    exp_dir1 = os.path.join("samples", exp_name)
    exp_dir2 = os.path.join("results", exp_name)

# ==============================================================================
#                              Validate configure
# ==============================================================================
if mode == "valid":
    # Additional variables.
    exp_dir    = os.path.join("results", "test", exp_name)

    # Load model.
    model      = Generator().to(device)
    model_path = args.weights

    # Test data address.
    lr_dir     = args.test_paths[0]
    sr_dir     = args.save_dir
    hr_dir     = args.test_paths[1]
