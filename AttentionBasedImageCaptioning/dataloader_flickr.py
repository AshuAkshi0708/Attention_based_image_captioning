import torch
import torchvision.transforms as transforms
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os
import pickle
import numpy as np
import nltk
from PIL import Image
import json as j
from torch.utils.data.sampler import SubsetRandomSampler
from vocab_builder_flickr import Vocabulary
#from pycocotools.coco import 


class randomsample(data.sampler.Sampler):
    def __init__(self,len_dataset,len_subset):
        self.len_dataset= len_dataset
        self.len_subset = len_subset
    def __iter__(self):
        return iter(np.random.permutation(np.arange(self.len_dataset,dtype = int))[:self.len_subset])
    def __len__(self):
        return self.len_subset

class FlickrDataset(data.Dataset):
    """Flickr Custom Dataset compatible with torch.utils.data.DataLoader."""
    def __init__(self, root, json, vocab, transform=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            root: image directory.
            json: flickr annotation file path.
            vocab: vocabulary wrapper.
            transform: image transformer.
        """
        self.root = root
        #self.coco = COCO(json)
        with open(json) as f:
            self.flickr = j.load(f)
        #self.flickr = js.loads(json)
        #self.ids = list(self.coco.anns.keys())
        self.vocab = vocab
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data pair (image and caption)."""
        flickr = self.flickr
        vocab = self.vocab
        #ann_id = self.ids[index]
        caption = flickr['comment'][index]
        img_path = flickr['image_name'][index]
        #path = coco.loadImgs(img_id)[0]['file_name']

        image = Image.open(os.path.join(self.root, img_path)).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def __len__(self):
        return len(self.flickr['image_name'])


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.
    Args:
        data: list of tuple (image, caption).
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.
    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, lengths

def get_loader(root, json, vocab, transform, batch_size, subset_size=6000, shuffle=False, num_workers=4):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    # COCO caption dataset
    flickr = FlickrDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform
                        )

    # Data loader for COCO dataset
    # This will return (images, captions, lengths) for each iteration.
    # images: a tensor of shape (batch_size, 3, 224, 224).
    # captions: a tensor of shape (batch_size, padded_length).
    # lengths: a list indicating valid length for each caption. length is (batch_size).
    subsetSampler = randomsample(len(flickr),subset_size)
    data_loader = torch.utils.data.DataLoader(dataset=flickr,
                                              sampler = subsetSampler,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader

def create_split_loaders(root,json,vocab,batch_size, transform, seed = 17, 
                         p_val=0.1, p_test=0.2, shuffle=True,
                         show_sample=False, extras={}):
    """ Creates the DataLoader objects for the training, validation, and test sets.

    Params:
    -------
    - batch_size: (int) mini-batch size to load at a time
    - seed: (int) Seed for random generator (use for testing/reproducibility)
    - transform: A torchvision.transforms object - transformations to apply to each image
                 (Can be "transforms.Compose([transforms])")
    - p_val: (float) Percent (as decimal) of dataset to use for validation
    - p_test: (float) Percent (as decimal) of the dataset to split for testing
    - shuffle: (bool) Indicate whether to shuffle the dataset before splitting
    - show_sample: (bool) Plot a mini-example as a grid of the dataset
    - extras: (dict)
        If CUDA/GPU computing is supported, contains:
        - num_workers: (int) Number of subprocesses to use while loading the dataset
        - pin_memory: (bool) For use with CUDA - copy tensors into pinned memory
                  (set to True if using a GPU)
        Otherwise, extras is an empty dict.

    Returns:
    --------
    - train_loader: (DataLoader) The iterator for the training set
    - val_loader: (DataLoader) The iterator for the validation set
    - test_loader: (DataLoader) The iterator for the test set
    """

    # Get create a ChestXrayDataset object
    dataset = FlickrDataset(root=root,
                       json=json,
                       vocab=vocab,
                       transform=transform
                        )

    # Dimensions and indices of training set
    dataset_size = len(dataset)
    all_indices = list(range(dataset_size))

    # Shuffle dataset before dividing into training & test sets
    if shuffle:
        np.random.seed(seed)
        np.random.shuffle(all_indices)

    # Create the validation split from the full dataset
    val_split = int(np.floor(p_val * dataset_size))
    train_ind, val_ind = all_indices[val_split :], all_indices[: val_split]

    # Separate a test split from the training dataset
    test_split = int(np.floor(p_test * len(train_ind)))
    train_ind, test_ind = train_ind[test_split :], train_ind[: test_split]
    np.savetxt("test_ind.txt",test_ind)

    # Use the SubsetRandomSampler as the iterator for each subset
    sample_train = SubsetRandomSampler(train_ind)
    sample_test = SubsetRandomSampler(test_ind)
    sample_val = SubsetRandomSampler(val_ind)

    num_workers = 0
    pin_memory = False
    # If CUDA is available
    if extras:
        num_workers = extras["num_workers"]
        pin_memory = extras["pin_memory"]

    # Define the training, test, & validation DataLoaders
    
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=sample_train, num_workers=num_workers, shuffle = False, collate_fn=collate_fn,
                              pin_memory=pin_memory)

    test_loader = DataLoader(dataset, batch_size=batch_size,
                             sampler=sample_test, num_workers=num_workers, shuffle = False,collate_fn=collate_fn,
                              pin_memory=pin_memory)

    val_loader = DataLoader(dataset, batch_size=batch_size,
                            sampler=sample_val, num_workers=num_workers,shuffle = False,collate_fn=collate_fn,
                              pin_memory=pin_memory)


    # Return the training, validation, test DataLoader objects
    return (train_loader, val_loader, test_loader)


if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    vocab = pickle.load(open("data/vocab_flickr.pkl", "rb"))

    embed_size = 256
    vocab_size = len(vocab)
    batch_size = 32

    root = "data/Flickr/Flickr_images/flickr30k_images/flickr30k_images"
    json = "data/data_ann_1.json"
    train_loader, val_loader, test_loader = create_split_loaders(root,json,vocab,batch_size, seed = 17, transform=transform,
                         p_val=0.1, p_test=0.2, shuffle=False,
                         show_sample=False, extras={})
    #train_loader = get_loader(train_images, train_captions, vocab, transform, batch_size)

    for minibatch_count, (images, captions, lengths) in enumerate(train_loader, 0):
        print(images.shape, captions.shape)
