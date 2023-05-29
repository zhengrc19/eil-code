import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset


class FetchDemoDataset(Dataset):
    """Fetch gym env dataset."""

    def __init__(self, root_dir, label_path, transform=None, window_len=1,
                 loss_type='mse', use_state=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            label_path (string): path to labels npy file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            window_len (int, optional): length of window of images to be
                concatenated while training.
        """
        self.loss_type = loss_type
        self.use_state = use_state
        self.root_dir = root_dir
        action_seq_lens = np.load(label_path, allow_pickle=True).item()
        if 'mse' in self.loss_type:
            self.labels = torch.tensor(action_seq_lens['actions']).float()
        else:
            self.labels = torch.tensor(np.where(action_seq_lens['actions'])[1])
        self.seq_lens = action_seq_lens['seq_lens']
        if self.use_state:
            self.states = torch.tensor(action_seq_lens['states']).float()
            self.state_dim = self.states.shape[1]
            assert self.states.shape[0] == self.labels.shape[0]
        self.cumm_lens = np.zeros(self.seq_lens.shape)
        self.win_len = window_len
        cumm_sum = 0
        for i in range(self.seq_lens.shape[0]):
            cumm_sum += self.seq_lens[i]
            self.cumm_lens[i] = cumm_sum
        # self.labels = torch.tensor(np.load(label_path)).float()
        self.transform = transform

    def __len__(self):
        return self.labels.shape[0]

    def induce_imgname(self, idx):
        """
        0 -> 0000/0000.jpg
        49 -> 0000/0049.jpg
        50 -> 0001/0000.jpg
        101 -> 0001/0051.jpg
        """
        i = 0
        if idx > self.cumm_lens[-1]:
            import ipdb
            ipdb.set_trace()
        while idx >= self.cumm_lens[i]:
            i += 1
        assert(idx < self.cumm_lens[i])
        if i == 0:
            frame = idx
        else:
            frame = idx - self.cumm_lens[i - 1]
        # if 'image' in self.input_type:
        return ('%04d/%04d.jpg' % (i, frame))
        # return i, frame

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        label = self.labels[idx]
        img_name = self.induce_imgname(idx)
        # img_path = os.path.join(self.root_dir, img_name)
        img_names = [img_name] + [self.induce_imgname(max(0, idx - i))
                                    for i in range(1, self.win_len)]
        for i, name in enumerate(img_names):
            if name[:4] != img_name[:4]:
                img_names[i] = img_names[i - 1]
        img_paths = [
            os.path.join(self.root_dir, name)
            for name in img_names
        ]
        images = [plt.imread(img) for img in img_paths]
        # print(images[0].shape)
        # print(images[0])
        images[0] = (images[0] * 255).astype('uint8')
        # image = plt.imread(img_name)
        # image = image.permute(2,0,1)
        if self.transform is not None:
            images = [self.transform(image) for image in images]
        images = torch.stack(images)
        if self.use_state:
            state = self.states[idx]
            return {"state": state, "images": images}, label
        return images, label


class FetchRobotDemoDataset(FetchDemoDataset):
    def __init__(self, root_dir, label_path, transform=None, window_len=1, loss_type='mse', use_state=False):
        super().__init__(root_dir, label_path, transform, window_len, loss_type, use_state)
    
    def induce_imgname(self, idx):
        i = 0
        if idx > self.cumm_lens[-1]:
            import ipdb
            ipdb.set_trace(context=10)
        while idx >= self.cumm_lens[i]:
            i += 1
        assert(idx < self.cumm_lens[i])
        if i == 0:
            frame = idx
        else:
            frame = idx - self.cumm_lens[i - 1]
        return ('%d/color/%d.png' % (i + 1, frame))
