import numpy as np
from utils import plot_images

import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler


def get_train_valid_loader(data_dir,
                           batch_size,
                           random_seed,
                           valid_size=0.1,
                           shuffle=True,
                           show_sample=False,
                           num_workers=4,
                           pin_memory=True,
                           mode='original'):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the MNIST dataset. A sample
    9x9 grid of the images can be optionally displayed.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
      In the paper, this number is set to 0.1.
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])

    # load dataset
    dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=trans
    )

    if mode == 'original':
        process = None
    elif mode == 'translated':
        process = translate
    elif mode == 'clutterd':
        process = clutter

    if process is not None:
        dataset.train_data = process(dataset.train_data)

    num_train = len(dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    # visualize some images
    if show_sample:
        sample_loader = torch.utils.data.DataLoader(
            dataset, batch_size=9, shuffle=shuffle,
            num_workers=num_workers, pin_memory=pin_memory
        )
        data_iter = iter(sample_loader)
        images, labels = data_iter.next()
        X = images.numpy()
        X = np.transpose(X, [0, 2, 3, 1])
        plot_images(X, labels)

    return (train_loader, valid_loader)


def get_test_loader(data_dir,
                    batch_size,
                    num_workers=4,
                    pin_memory=False,
                    mode='original'):
    """
    Utility function for loading and returning a multi-process
    test iterator over the MNIST dataset.

    If using CUDA, num_workers should be set to 1 and pin_memory to True.

    Args
    ----
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.

    Returns
    -------
    - data_loader: test set iterator.
    """
    # define transforms
    normalize = transforms.Normalize((0.1307,), (0.3081,))
    trans = transforms.Compose([
        transforms.ToTensor(), normalize,
    ])

    # load dataset
    dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=trans
    )

    if mode == 'original':
        process = None
    elif mode == 'translated':
        process = translate
    elif mode == 'clutterd':
        process = clutter

    if process is not None:
        dataset.test_data = process(dataset.test_data)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader


def translate(batch):
    batch1 = batch.cpu().data.unsqueeze(1).numpy()
    n, c, w_i = batch1.shape[:3]
    w_o = 60
    data = np.zeros(shape=(n, c, w_o, w_o), dtype=np.float32)
    for k in range(n):
        i, j = np.random.randint(0, w_o - w_i, size=2)
        data[k, :, i:i + w_i, j:j + w_i] += batch1[k]
    return torch.from_numpy(data).type_as(batch).squeeze(1)


def clutter(batch):
    batch1 = batch.cpu().data.unsqueeze(1).numpy()
    n, c, w_i = batch1.shape[:3]
    w_o = 60
    data = np.zeros(shape=(n, c, w_o, w_o), dtype=np.float32)
    for k in range(n):
        i, j = np.random.randint(0, w_o - w_i, size=2)
        data[k, :, i:i + w_i, j:j + w_i] += batch1[k]
        for _ in range(4):
            clt = batch1[np.random.randint(0, batch1.shape[0] - 1)]
            c1, c2 = np.random.randint(0, w_i - 8, size=2)
            i1, i2 = np.random.randint(0, w_o - 8, size=2)
            data[k, :, i1:i1 + 8, i2:i2 + 8] += clt[:, c1:c1 + 8, c2:c2 + 8]
    data = np.clip(data, 0., 1.)
    return torch.from_numpy(data).type_as(batch).squeeze(1)
