import os, argparse, time, json, sys
sys.path.append('..')
import numpy as np
from tqdm import tqdm
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as trn
import torchvision.datasets as dset
from torchvision.models import resnet50

from utils import (
    FinegrainedDataset, SPLIT_NUM_CLASSES, INET_SPLITS, WebVision,
    AverageMeter, accuracy
)


# /////////////// Setup ///////////////
# Arguments
parser = argparse.ArgumentParser(description='Trains a classifier')
# Dataset options
parser.add_argument('--dataset', type=str, choices=['bird', 'butterfly', 'car', 'aircraft'],
                    help='Choose the dataset', required=True)
parser.add_argument('--data-dir', type=str, default='../data')
parser.add_argument('--info-dir', type=str, default='../info')
parser.add_argument('--split', '-s', type=int, default=0)
# Model options
parser.add_argument('--model', '-m', type=str, default='rn', help='Choose architecture.')
# OE options
parser.add_argument('--beta', type=float, default=1.0, help='Weighting factor for the OE objective.')
parser.add_argument('--oe-set', type=str, default='WebVision', choices=['WebVision'])
# Optimization options
parser.add_argument('--torch-seed', '-ts', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', '-e', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='The initial learning rate.')
parser.add_argument('--batch-size', '-b', type=int, default=32, help='Batch size.')
parser.add_argument('--test-bs', type=int, default=100)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.00001, help='Weight decay (L2 penalty).')
# Checkpoints
parser.add_argument('--save-dir', type=str, default=None, help='Folder to save checkpoints.')
# Acceleration
parser.add_argument('--gpu', nargs='*', type=int, default=[0,1])
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()


DATA_DIR = {
    'bird': os.path.join(args.data_dir, 'bird', 'images'),
    'butterfly': os.path.join(args.data_dir, 'butterfly', 'images_small'),
    'car': os.path.join(args.data_dir, 'car'),
    'aircraft': os.path.join(args.data_dir, 'aircraft', 'images')
}

# Set random seed for torch
torch.manual_seed(args.torch_seed)

# Create ID/OOD splits
all_classes = list(range(sum(SPLIT_NUM_CLASSES[args.dataset])))
rng = np.random.RandomState(args.split)
id_classes = list(rng.choice(all_classes, SPLIT_NUM_CLASSES[args.dataset][0], replace=False))
ood_classes = [c for c in all_classes if c not in id_classes]
print(f'# ID classes: {len(id_classes)}, # OOD classes: {len(ood_classes)}')

# Datasets and dataloaders
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

train_transform = trn.Compose([
    trn.Resize((512, 512)),
    trn.RandomCrop((448, 448)),
    trn.RandomHorizontalFlip(),
    trn.ColorJitter(brightness=32./255., saturation=0.5),
    trn.ToTensor(),
    trn.Normalize(mean, std),
])
test_transform = trn.Compose([
    trn.Resize((512, 512)),
    trn.CenterCrop(448),
    trn.ToTensor(),
    trn.Normalize(mean, std),
])

train_set = FinegrainedDataset(
    image_dir=DATA_DIR[args.dataset], 
    info_filepath=os.path.join(args.info_dir, args.dataset, 'train.txt'), 
    class_list=id_classes,
    transform=train_transform
)
test_set = FinegrainedDataset(
    image_dir=DATA_DIR[args.dataset], 
    info_filepath=os.path.join(args.info_dir, args.dataset, 'val.txt'), 
    class_list=id_classes,
    transform=test_transform
)
num_classes = train_set.num_classes
assert num_classes == len(id_classes)
print(f'Train samples: {len(train_set)}, Val samples: {len(test_set)}')

if args.oe_set == 'WebVision':
    class_to_be_removed = []
    for l in INET_SPLITS.values():
        class_to_be_removed.extend(l)
    oe_set = WebVision(
        root='../data/WebVision', transform=train_transform,
        concept_list=class_to_be_removed, exclude=True
    )
else:
    raise NotImplementedError

train_loader = DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, 
    num_workers=args.prefetch, pin_memory=False, drop_last=True
)
test_loader = DataLoader(
    test_set, batch_size=args.test_bs, shuffle=False, 
    num_workers=args.prefetch, pin_memory=False
)

# Set up checkpoint directory and tensorboard writer
if args.save_dir is None:
    oe_related = f'oem_{args.oe_set}'
    oe_related += f'_beta={args.beta}'

    args.save_dir = os.path.join(
        '../checkpoints', args.dataset, 
        f'split_{args.split}',
        f'{args.model}_{oe_related}_epochs={args.epochs}_bs={args.batch_size}'
    )
else:
    assert 'checkpoints' in args.save_dir, \
        "If 'checkpoints' not in save_dir, then you may have an unexpected directory for writer..."
chkpnt_path = os.path.join(args.save_dir, f'seed_{args.torch_seed}.pth')

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
elif os.path.isfile(chkpnt_path):
    print('*********************************')
    print('* The checkpoint already exists *')
    print('*********************************')

writer = SummaryWriter(args.save_dir.replace('checkpoints', 'runs'))

# Set up GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(lambda x: str(x), args.gpu))

# Create model
if args.model == 'rn':
    net = resnet50()
    pretrained_model_file = f'../checkpoints/{args.dataset}/split_{args.split}/rn_baseline_epochs=90_bs=32/seed_0.pth'
    state_dict = torch.load(pretrained_model_file)
else:
    raise NotImplementedError

in_features = net.fc.in_features
new_fc = nn.Linear(in_features, num_classes)
net.fc = new_fc

try:
    net.load_state_dict(state_dict)
except RuntimeError:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.` caused by nn.DataParallel
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

net.cuda()
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)
cudnn.benchmark = True  # fire on all cylinders

# Optimizer and scheduler
optimizer = optim.SGD(
    net.parameters(), args.lr, momentum=args.momentum,
    weight_decay=args.decay, nesterov=True
)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

scheduler = optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.lr)
)

# /////////////// Training ///////////////
# train function
def train():
    # select hardest outliers for training
    net.eval()

    # here we only select from a random group of 50000 samples
    # otherwise this step could take very long since the OE set
    # is pretty large
    temp_oe_set = Subset(oe_set, rng.choice(range(len(oe_set)), 50000, replace=False))
    oe_temp_loader = DataLoader(
        temp_oe_set, 
        batch_size=200, shuffle=False, num_workers=args.prefetch, pin_memory=False
    )

    oe_ood_score = []
    with torch.no_grad():
        for x, _ in tqdm(oe_temp_loader, total=len(oe_temp_loader), leave=False):
            x = x.cuda()
            output = F.softmax(net(x), dim=-1)
            oe_ood_score.append(
                output.max(1)[0].cpu()
            )

    oe_ood_score = torch.cat(oe_ood_score)
    assert len(oe_ood_score) == len(temp_oe_set)

    num_oe = 2 * args.batch_size * (len(train_set) // args.batch_size)
    oe_loader = DataLoader(
        Subset(temp_oe_set, torch.topk(oe_ood_score, num_oe, largest=True)[1]),
        batch_size=2*args.batch_size, shuffle=True, pin_memory=False
    )

    # training
    net.train()  # enter train mode

    current_lr = scheduler.get_last_lr()[0]
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    batch_iter = tqdm(train_loader, total=len(train_loader), 
        desc='Batch', leave=False, position=2)

    oe_iter = iter(oe_loader)

    for x, y in batch_iter:
        batch_size = x.size(0)
        try:
            oe_x, _ = next(oe_iter)
        except StopIteration:
            continue
        y = y.cuda()

        data = torch.cat((x, oe_x)).cuda()
        logits = net(data)
        loss = F.cross_entropy(logits[:batch_size], y)

        # cross-entropy from softmax distribution to uniform distribution
        loss += args.beta * -(logits[batch_size:].mean(1) - torch.logsumexp(logits[batch_size:], dim=1)).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        acc1, acc5 = accuracy(logits[:batch_size], y, topk=(1, 5))
        losses.update(loss.item(), x.size(0))
        top1.update(acc1, x.size(0))
        top5.update(acc5, x.size(0))
    
    print_message = f'Epoch [{epoch:3d}] | Loss: {losses.avg:.4f}, ' \
        f'Top1 Acc: {top1.avg:.2f}, Top5 Acc: {top5.avg:.2f}'
    tqdm.write(print_message)

    writer.add_scalar('train/loss', losses.avg, epoch)
    writer.add_scalar('train/acc_top1', top1.avg, epoch)
    writer.add_scalar('train/acc_top5', top5.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)

# test function
def test():
    net.eval()

    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            output = net(x)
            
            acc1, acc5 = accuracy(output, y, topk=(1, 5))
            top1.update(acc1, x.size(0))
            top5.update(acc5, x.size(0))
    
    print_message = f'Evaluation  | Top1 Acc: {top1.avg:.2f}, Top5 Acc: {top5.avg:.2f}\n'
    tqdm.write(print_message)

    writer.add_scalar('test/acc_top1', top1.avg, epoch)
    writer.add_scalar('test/acc_top5', top5.avg, epoch)

    return top1.avg

# Main loop
epoch_iter = tqdm(list(range(1, args.epochs+1)), total=args.epochs, desc='Epoch',
    leave=True, position=1)
    
best_acc1 = 0
for epoch in epoch_iter:
    train()
    acc1 = test()

    if acc1 > best_acc1:
        # Save model
        torch.save(
            net.state_dict(),
            chkpnt_path
        )
    best_acc1 = max(acc1, best_acc1)