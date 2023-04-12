import os, argparse, time, json, datetime, sys, pickle, random
sys.path.append('..')
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from skimage.filters import gaussian as gblur
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as trn
import torchvision.datasets as dset
from torchvision.models import resnet50

from utils import (
    silence_PIL_warnings, FinegrainedDataset, 
    print_measures_with_std, print_measures,
    SPLIT_NUM_CLASSES
)
import utils.calculate_log as callog

# EXIF warning silent
silence_PIL_warnings()

concat = lambda x: np.concatenate(x, axis=0)
to_np = lambda x: x.data.cpu().numpy()


def get_baseline_scores(model, loader, in_dist=False, max_logits=False, sum_logits=False, energy=False, temp=1):
    assert sum([max_logits, sum_logits, energy]) <= 1

    _score = []
    total = 0
    correct = []

    with torch.no_grad():
        for batch_idx, (data, target) in tqdm(enumerate(loader), total=len(loader)):
            data = data.cuda()
            output = model(data)

            if in_dist:
                correct.append(output.argmax(1).cpu().eq(target))
            
            if max_logits:
                _score.append(np.max(to_np(output), axis=1))

            if sum_logits:
                _score.append(to_np(torch.sum(output, dim=1)))
            
            if energy:
                _score.append(to_np((temp*torch.logsumexp(output/temp, dim=1))))
            
            if not (max_logits or sum_logits or energy):
                smax = to_np(F.softmax(output, dim=1))
                _score.append(np.max(smax, axis=1))    
            
            total += data.size(0)

    if in_dist:
        return concat(_score).copy(), torch.cat(correct).numpy() #acc/total
    else:
        return concat(_score).copy()


def get_odin_scores(model, loader, in_dist=False, temp=1000):
    _score = []
    acc = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(loader):
            data = data.cuda()
            output = model(data)

            if in_dist:
                acc += output.argmax(1).eq(target.cuda()).sum().item()

            # temperature scaling
            output = output / temp
            smax = to_np(F.softmax(output, dim=1))
            _score.append(np.max(smax, axis=1))
        
            total += data.size(0)

    if in_dist:
        return concat(_score).copy(), acc/total
    else:
        return concat(_score).copy()


def get_and_print_results(in_score, model, ood_loader, ood_fn, num_run=10, **kwargs):
    out_score = ood_fn(model, ood_loader, **kwargs)
    sample_size = min(len(in_score), len(out_score))

    aurocs, auprs, tnrs = [], [], []
    random.seed(0)
    for i in range(num_run):
        metric_results = callog.metric(
            np.array(random.sample(list(in_score), sample_size)),
            np.array(random.sample(list(out_score), sample_size))
        )
        # update stat keepers for auroc, tnr, aupr
        aurocs.append(metric_results['TMP']['AUROC'])
        tnrs.append( metric_results['TMP']['TNR'])
        auprs.append(metric_results['TMP']['AUIN'])

    auroc = np.mean(aurocs); aupr = np.mean(auprs); tnr = np.mean(tnrs)

    if num_run >= 1:
        msg = print_measures_with_std(aurocs, auprs, tnrs)
    else:
        msg = print_measures(auroc, aupr, tnr)

    if args.ood == 'energy' and args.val:
        return auroc, aupr, tnr, msg, np.mean(out_score)
    else:
        return auroc, aupr, tnr, msg
    

def print_and_write(msg, fp):
    if isinstance(msg, str):
        print(msg)
        if fp is not None: fp.write(msg+'\n')
    elif isinstance(msg, list):
        print('\n'.join(msg))
        if fp is not None: fp.write('\n'.join(msg))
        if fp is not None: fp.write('\n')
    else:
        raise TypeError


# /////////////// Setup ///////////////
# Arguments
parser = argparse.ArgumentParser(description='Evaluates OOD detection of a classifier')
parser.add_argument('--data-dir', type=str, default='../data')
parser.add_argument('--info-dir', type=str, default='../info')
# Model loading
parser.add_argument('--model-file', type=str, required=True)
# General OOD detection options
parser.add_argument('--batch-size', '-b', type=int, default=100, help='Batch size.')
# OOD options
parser.add_argument('--ood', type=str, default='msp',
    choices=['msp', 'maxlogits', 'sumlogits', 'odin', 'energy'])
parser.add_argument('--temp', type=float, default=1)
parser.add_argument('--val', action='store_true')
# Results saving options
parser.add_argument('--save-to-file', '-s', action='store_true')
parser.add_argument('--overwrite', '-o', action='store_true')
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

args.model_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'checkpoints', args.model_file)
assert os.path.isfile(args.model_file)

temp = args.model_file.split('/')
dataset = temp[temp.index('checkpoints')+1]
split_idx = int(temp[temp.index('checkpoints')+2].split('_')[-1])
arch = args.model_file.split('/')[-2].split('_')[0]

print(f'Dataset: {dataset}, Split ID: {split_idx}, Arch: {arch}')
    
# Set up txt file for recording results
if args.save_to_file:
    result_root = '/'.join(args.model_file.replace('checkpoints', 'results').split('/')[:-1])
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    
    prefix = args.model_file.split('/')[-1].split('.')[0]
    if args.val: prefix = 'val_' + prefix

    if args.ood in ['odin', 'energy']:
        filename = '%s_%s_temp_%g.txt' % (prefix, args.ood, args.temp)
    else:
        filename = '%s_%s.txt' % (prefix, args.ood)

    result_filepath = os.path.join(result_root, filename)
    if os.path.isfile(result_filepath) and not args.overwrite:
        f = open(result_filepath, 'a')
    else:
        f = open(result_filepath, 'w')
    f.write('\n\nTime: %s\n' % str(datetime.datetime.now()))
else:
    f = None


# Create ID/OOD splits
all_classes = list(range(sum(SPLIT_NUM_CLASSES[dataset])))
rng = np.random.RandomState(split_idx)
id_classes = list(rng.choice(all_classes, SPLIT_NUM_CLASSES[dataset][0], replace=False))
ood_classes = [c for c in all_classes if c not in id_classes]

# Datasets and dataloaders
## In-distribution
mean=[0.485, 0.456, 0.406]
std=[0.229, 0.224, 0.225]

test_transform = trn.Compose([
    trn.Resize((512, 512)),
    trn.CenterCrop(448),
    trn.ToTensor(),
    trn.Normalize(mean, std),
])

if args.val:
    id_set = FinegrainedDataset(
        image_dir=DATA_DIR[dataset], 
        info_filepath=os.path.join(args.info_dir, dataset, 'val.txt'), 
        class_list=id_classes,
        transform=test_transform
    )
else:
    id_set = FinegrainedDataset(
        image_dir=DATA_DIR[dataset], 
        info_filepath=os.path.join(args.info_dir, dataset, 'test.txt'), 
        class_list=id_classes,
        transform=test_transform
    )

id_loader = DataLoader(
    id_set, batch_size=args.batch_size, shuffle=False, 
    num_workers=args.prefetch
)
id_num_examples = len(id_set)
num_classes = id_set.num_classes
assert num_classes == len(id_classes)

## Out-of-distribution
if args.val:
    class_to_be_removed = []
    for l in INET_SPLITS.values():
        class_to_be_removed.extend(l)

    ood_sets = {}
    temp = WebVision(
        root='../data/WebVision', transform=test_transform,
        concept_list=class_to_be_removed, exclude=True
    )
    random.seed(0)
    ood_sets['WebVision'] = Subset(temp, random.sample(range(len(temp)), len(id_set)))
else:
    ood_sets = {}

    others = [k for k in DATA_DIR.keys() if k != dataset]
    # combine other datasets into a single dataset
    for i, dset_name in enumerate(others):
        if i == 0:
            others_set = FinegrainedDataset(
                image_dir=DATA_DIR[dset_name], 
                info_filepath=os.path.join(args.info_dir, dset_name, 'test.txt'), 
                transform=test_transform
            )
        else:
            temp = FinegrainedDataset(
                image_dir=DATA_DIR[dset_name], 
                info_filepath=os.path.join(args.info_dir, dset_name, 'test.txt'), 
                transform=test_transform
            )
            others_set.samples.extend(temp.samples)
    random.seed(0)
    others_set.samples = random.sample(others_set.samples, len(id_set))
    ood_sets['coarse'] = others_set

    # fine-grained hold-out set
    holdout_set = FinegrainedDataset(
        image_dir=DATA_DIR[dataset], 
        info_filepath=os.path.join(args.info_dir, dataset, 'test.txt'), 
        class_list=ood_classes,
        transform=test_transform
    )
    ood_sets['fine'] = holdout_set

ood_loaders = {
    k: DataLoader(
        v, batch_size=args.batch_size, shuffle=True,
        num_workers=args.prefetch
    ) for k, v in ood_sets.items()
}

# Set up GPU
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(lambda x: str(x), args.gpu))

# Get model
if arch == 'rn':
    net = resnet50()
else:
    raise NotImplementedError

in_features = net.fc.in_features
new_fc = nn.Linear(in_features, num_classes)
net.fc = new_fc
if 'rot' in args.model_file:
    net.rot_head = nn.Linear(in_features, 4)

state_dict = torch.load(args.model_file)
try:
    net.load_state_dict(state_dict)
except RuntimeError:
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.` caused by nn.DataParallel
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

net.eval()
net.cuda()
if torch.cuda.device_count() > 1:
    net = torch.nn.DataParallel(net)

# OOD configuration
if args.ood in ['msp', 'maxlogits', 'sumlogits','energy']:
    ood_kwargs = {
        'max_logits': args.ood == 'maxlogits',
        'sum_logits': args.ood == 'sumlogits',
        'energy': args.ood == 'energy',
        'temp': args.temp
    }
    ood_fn = get_baseline_scores
elif args.ood == 'odin':
    ood_kwargs = {'temp': args.temp}
    ood_fn = get_odin_scores

# Get ID scores
in_score, in_correct = ood_fn(net, id_loader, in_dist=True, **ood_kwargs)
acc = np.mean(in_correct)
print_and_write([
    'Acc:         %.2f' % (100.*acc) 
], f)

if args.ood == 'energy' and args.val:
    print_and_write([
        'In_energy:  %.2f' % np.mean(in_score)
    ], f)

# Get OOD scores
for ood_name, ood_loader in ood_loaders.items():
    if args.ood == 'energy' and args.val:
        auroc, aupr, tnr, msg, out_energy = \
            get_and_print_results(
                in_score, net, ood_loader,
                ood_fn, **ood_kwargs
            )
    else:
        auroc, aupr, tnr, msg = get_and_print_results(
            in_score, net, ood_loader,
            ood_fn, **ood_kwargs
        )

    print_and_write(['\n', ood_name], f)
    print_and_write(msg, f)
    if args.ood == 'energy' and args.val:
        print_and_write([
            'Out_energy:  %.2f' % out_energy
        ], f)
    
if f is not None:
    f.write('='*50)
    f.close()
