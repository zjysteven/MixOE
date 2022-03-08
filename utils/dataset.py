import os, random, pickle
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset, ConcatDataset
from torchvision.datasets.folder import default_loader, make_dataset
from torchvision.datasets.vision import VisionDataset


SPLIT_NUM_CLASSES = {
    'bird': [200, 55],
    'butterfly': [150, 50],
    'car': [150, 46],
    'aircraft': [90, 10]
}


IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

INET_SPLITS = {
    'dog': [
        'n02089078', 'n02102318', 'n02096294', 'n02099712', 'n02112018', 'n02113023', 'n02110063', 'n02099601', 'n02097658', 'n02115641', 'n02112350', 'n02106166', 'n02101556', 'n02111500', 'n02106382', 'n02108422', 'n02093859', 'n02098286', 'n02107574', 'n02108551', 'n02088094', 'n02097047', 'n02113799', 'n02093428', 'n02085936', 'n02100735', 'n02099849', 'n02096051', 'n02089973', 'n02101006',
        'n02088632', 'n02093754', 'n02095314', 'n02088466', 'n02086240', 'n02110185', 'n02087046', 'n02091831', 'n02105412', 'n02105641', 'n02094114', 'n02107683', 'n02106550', 'n02111129', 'n02113186', 'n02100583', 'n02111889', 'n02108915', 'n02102973', 'n02097298', 'n02104029', 'n02090622', 'n02090379', 'n02096585', 'n02092002', 'n02087394', 'n02092339', 'n02091467', 'n02110806', 'n02096177',
        'n02089867', 'n02099267', 'n02093647', 'n02088364', 'n02095570', 'n02085782', 'n02105056', 'n02109525', 'n02115913', 'n02112706', 'n02085620', 'n02095889', 'n02101388', 'n02109961', 'n02086646', 'n02102177', 'n02093256', 'n02111277', 'n02098413', 'n02088238', 'n02108000', 'n02097474', 'n02113978', 'n02086079', 'n02091032', 'n02106030', 'n02093991', 'n02102040', 'n02113712', 'n02090721',
        'n02109047', 'n02100236', 'n02102480', 'n02099429', 'n02097130', 'n02108089', 'n02105505', 'n02112137', 'n02094258', 'n02110627', 'n02105162', 'n02113624', 'n02106662', 'n02091635', 'n02086910', 'n02098105', 'n02096437', 'n02091244', 'n02116738', 'n02104365', 'n02107908', 'n02105855', 'n02091134', 'n02100877', 'n02097209', 'n02094433', 'n02107142', 'n02110958', 'n02107312', 'n02105251'
    ],
    'bird': [
        'n01514668', 'n02051845', 'n02018207', 'n01531178', 'n01828970', 'n01514859', 'n01558993', 'n02056570', 'n02018795', 'n01580077', 'n01847000', 'n02013706', 'n01614925',
        'n02028035', 'n02037110', 'n02011460', 'n01855672', 'n02058221', 'n01534433', 'n01616318', 'n01855032', 'n01518878', 'n01537544', 'n02002556', 'n01843383', 'n01843065',
        'n01622779', 'n01860187', 'n01820546', 'n01530575', 'n01819313', 'n01608432', 'n02033041', 'n01818515', 'n02009912', 'n02002724', 'n01532829', 'n01560419', 'n01601694',
        'n02012849', 'n01582220', 'n01824575', 'n01829413', 'n02027492', 'n01592084', 'n02025239', 'n02006656', 'n02007558', 'n02017213', 'n01833805', 'n01817953', 'n02009229'
    ],
    'car': [
        'n04467665', 'n03895866', 'n04285008', 'n02704792', 'n04465501', 'n03272562', 'n03384352',
        'n03345487', 'n02797295', 'n04252077', 'n03777568', 'n03791053', 'n03478589', 'n03930630',
        'n03977966', 'n03444034', 'n03792782', 'n04335435', 'n04204347', 'n02930766', 'n03670208',
        'n04252225', 'n03100240', 'n03785016', 'n03417042', 'n03538406', 'n04065272', 'n04310018',
        'n04461696', 'n04509417', 'n03770679', 'n03594945', 'n04389033', 'n03868242', 'n04482393',
        'n04037443', 'n02814533', 'n03599486', 'n03393912', 'n02701002', 'n03796401', 'n02835271',
    ],
    'butterfly': [
        'n02277742', 'n02281787', 'n02279972', 'n02281406', 'n02280649', 'n02276258'
    ],
    'aircraft': [
        'n02690373', 'n02692877', 'n02782093',
    ]
}


####################################################
# Custom dataset for loading fine-grained datasets #
####################################################
class FinegrainedDataset(Dataset):
    def __init__(self, image_dir, info_filepath, class_list=None, transform=None, target_transform=None):
        super(FinegrainedDataset, self).__init__()

        self.dset_name = info_filepath.split('/')[-2]

        all_imagepath = []
        all_labels = []
        with open(info_filepath, 'r') as f:
            for l in f.readlines():
                line = l.rstrip()
                if len(line.split()) == 2:
                    img_name = line.split()[0]
                    label = int(line.split()[1])
                elif len(line.split()) > 2:
                    img_name = ' '.join(line.split()[:2])
                    label = int(line.split()[2])
                else:
                    raise RuntimeError

                all_imagepath.append(img_name)
                all_labels.append(label)
        
        if not class_list:
            class_to_id = {k: k for k in sorted(set(all_labels))}
        else:
            class_to_id = {k: i for i, k in enumerate(sorted(class_list))}
        self.class_to_id = class_to_id

        self.samples = [
            [os.path.join(image_dir, image_path), class_to_id[l]] \
                for (image_path, l) in zip(all_imagepath, all_labels) if class_to_id.get(l) is not None
        ]
        
        self.transform = transform
        self.target_transform = target_transform
        self.loader = default_loader
        self.num_classes = len(class_to_id)

    def __getitem__(self, index):
        path, target = self.samples[index]
        
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


####################################################
# Below is for loading ImageNet and WebVision      #
####################################################
class DatasetFolder(VisionDataset):
    """A generic data loader where the samples are arranged in this way: ::
        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/[...]/xxz.ext
        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/[...]/asd932_.ext
    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (tuple[string]): A list of allowed extensions.
            both extensions and is_valid_file should not be passed.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.
        is_valid_file (callable, optional): A function that takes path of a file
            and check if the file is a valid file (used to check of corrupt files)
            both extensions and is_valid_file should not be passed.
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(
        self, 
        root, 
        loader, 
        extensions=None, 
        transform=None, 
        target_transform=None,
        is_valid_file=None,
        class_list=None,
        exclude=False,
        num_classes=None
    ):
        super(DatasetFolder, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.class_list = class_list
        self.exclude = exclude
        self.num_classes = num_classes
        classes, class_to_idx = self._find_classes(self.root)
        samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)
        if len(samples) == 0:
            msg = "Found 0 files in subfolders of: {}\n".format(self.root)
            if extensions is not None:
                msg += "Supported extensions are: {}".format(",".join(extensions))
            raise RuntimeError(msg)

        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]
    
    @staticmethod
    def make_dataset(
        directory,
        class_to_idx,
        extensions=None,
        is_valid_file=None,
    ):
        return make_dataset(directory, class_to_idx, extensions=extensions, is_valid_file=is_valid_file)

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if self.class_list is None:
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            if self.exclude:
                classes = [d.name for d in os.scandir(dir) if (d.is_dir() and (d.name not in self.class_list))]
            else:
                classes = [d.name for d in os.scandir(dir) if (d.is_dir() and (d.name in self.class_list))]
        
        if self.num_classes:
            assert self.num_classes <= len(classes)
            classes = list(np.random.choice(classes, self.num_classes, replace=False))
        
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/[...]/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/[...]/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        is_valid_file (callable, optional): A function that takes path of an Image file
            and check if the file is a valid file (used to check of corrupt files)
     Attributes:
        classes (list): List of the class names sorted alphabetically.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(
            self,
            root,
            transform=None,
            target_transform=None,
            loader=default_loader,
            is_valid_file=None,
            class_list=None,
            exclude=False,
            num_classes=None
    ):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS if is_valid_file is None else None,
                                          transform=transform,
                                          target_transform=target_transform,
                                          is_valid_file=is_valid_file,
                                          class_list=class_list,
                                          exclude=exclude,
                                          num_classes=num_classes)
        self.imgs = self.samples


def WebVision(root, transform=None, target_transform=None, concept_list=None, exclude=False, num_concepts=None):
    # first settle down the concepts (classes of ImageNet)
    all_concept_wnid_to_name = {}
    all_concept_wnid_to_idx = {}
    with open(os.path.join(root, 'info', 'synsets.txt'), 'r') as f:
        for i, line in enumerate(f):
            temp = line.rstrip()
            all_concept_wnid_to_name[temp[:9]] = temp[10:]
            all_concept_wnid_to_idx[temp[:9]] = i+1

    if concept_list is not None:
        if exclude:
            _concept_list = list(filter(lambda x: x not in concept_list, list(all_concept_wnid_to_name.keys())))
        else:
            _concept_list = list(filter(lambda x: x in concept_list, list(all_concept_wnid_to_name.keys())))
    else:
        _concept_list = list(all_concept_wnid_to_name.keys())

    if num_concepts:
        _concept_list = np.random.choice(_concept_list, num_concepts, replace=False)
    _num_concepts = len(_concept_list)

    # then get corresponding queries
    all_concept_idx_to_query_idx = {i: [] for i in range(1, len(all_concept_wnid_to_name)+1)}
    with open(os.path.join(root, 'info', 'queries_synsets_map.txt'), 'r') as f:
        for line in f:
            temp = line.rstrip()
            all_concept_idx_to_query_idx[int(temp.split(' ')[1])].append(int(temp.split(' ')[0]))
        
    query_list = []
    for concept in _concept_list:
        concept_idx = all_concept_wnid_to_idx[concept]
        query_list.extend(
            list(map(
                lambda x: 'q'+str(x).zfill(4),
                all_concept_idx_to_query_idx[concept_idx]
            ))
        )
        
    google_set = ImageFolder(
        os.path.join(root, 'google'), class_list=query_list, transform=transform, target_transform=target_transform
    )

    flickr_set = ImageFolder(
        os.path.join(root, 'flickr'), class_list=query_list, transform=transform, target_transform=target_transform
    )

    return ConcatDataset([google_set, flickr_set])



if __name__ == '__main__':
    class_to_be_removed = []
    for l in INET_SPLITS.values():
        class_to_be_removed.extend(l)
    print(len(class_to_be_removed))