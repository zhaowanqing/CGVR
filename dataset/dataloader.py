import json
import torch.utils.data as data
import nltk
import numpy as np
from torchvision import transforms
import torch.utils.data
from PIL import Image
import torch
from nltk.tokenize import MWETokenizer


def default_loader(path):
    return Image.open(path).convert('RGB')


def getTagsList(tagsListPath):
    f = open(tagsListPath, encoding='utf-8')
    lines = f.readlines()
    tagsList = []
    for line in lines:
        line = line.strip().strip("\n")
        tagsList.append(line)
    f.close()
    return tagsList


def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x


class PreDataset(data.Dataset):
    """
    nuswide, flickr25k
    """

    def __init__(self, base_path, data_path, vocab, transform=None,
                 loader=default_loader):
        self.vocab = vocab
        loc = base_path + '/'
        self.tokenizer = MWETokenizer([("<", "pad", ">")], "")

        with open(loc + data_path, 'r', encoding='utf-8') as fh:
            singletons = []
            for line in fh:
                line = line.strip('\n')
                line = line.rstrip('\n')
                words = line.split('  ')
                singletons.append((loc + words[0], words[1].lower()))
                self.singletons = singletons
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        fn, caption = self.singletons[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        vocab = self.vocab
        tokens = self.tokenizer.tokenize(nltk.tokenize.word_tokenize(caption))
        caption = []
        caption.extend([vocab(token) for token in tokens])
        target = torch.Tensor(caption).to(torch.int32)
        return img, target, index

    def __len__(self):

        return len(self.singletons)


def collate_fn(data):
    images, captions, ids = zip(*data)
    data = [item[0] for item in data]
    images = torch.stack(data, 0)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]
    return images, targets, ids, lengths


def deserialize_vocab(src):
    with open(src) as f:
        d = json.load(f)
    vocab = Vocabulary()
    vocab.word2idx = d['word2idx']
    vocab.idx2word = d['idx2word']
    vocab.idx = d['idx']
    return vocab


class Vocabulary(object):
    """Simple vocabulary wrapper."""

    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if word not in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


def serialize_vocab(vocab, dest):
    d = {}
    d['word2idx'] = vocab.word2idx
    d['idx2word'] = vocab.idx2word
    d['idx'] = vocab.idx
    with open(dest, "w") as f:
        json.dump(d, f)


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])
valid_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize
])


def get_train_loaders(base_path, data_path, vocab, batch_size, workers):
    dset = PreDataset(base_path, data_path, vocab, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(dataset=dset, batch_size=batch_size, shuffle=True,
                                               pin_memory=False, collate_fn=collate_fn, num_workers=workers)
    return train_loader


def get_base_loaders(base_path, data_path, vocab, batch_size, workers):
    dset = PreDataset(base_path, data_path, vocab, transform=valid_transform)
    base_loader = torch.utils.data.DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                              collate_fn=collate_fn, num_workers=workers)
    return base_loader


def get_query_loaders(base_path, data_path, vocab, batch_size, workers):
    dset = PreDataset(base_path, data_path, vocab, transform=valid_transform)
    query_loader = torch.utils.data.DataLoader(dataset=dset, batch_size=batch_size, shuffle=False, pin_memory=False,
                                               collate_fn=collate_fn, num_workers=workers)
    return query_loader

