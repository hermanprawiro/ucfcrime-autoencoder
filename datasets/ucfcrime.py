import os
import torch
import torch.utils.data as torch_data
from PIL import Image

class UCFCrime(torch_data.Dataset):
    def __init__(self, root_dir, train=True, transforms=None, clip_size=16, clip_stride=1, clip_dilation=0):
        self.root_dir = root_dir
        self.train = train
        self.transforms = transforms

        self.clip_size = clip_size
        self.clip_stride = clip_stride
        self.clip_dilation = clip_dilation
        self.clip_dilated_frames_size = self._getdilatedsize()

        self.list = self._getlist()

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        img_path, label, idx = self.list[index]
        indices = self._getindices(idx)
        img_path = os.path.join(self.root_dir, img_path)
        img_file_template = '{:06d}.jpg'

        imgs = [Image.open(os.path.join(img_path, img_file_template.format(i + 1))) for i in indices]
        if self.transforms is not None:
            imgs = self.transforms(imgs)

        return imgs, label

    def _getlist(self):
        annotation_file = 'train_list_road.txt' if self.train else 'test_list_road.txt'
        annotation_lists = [row.strip('\n') for row in open(os.path.join(os.path.dirname(__file__), annotation_file), 'r')]
        lists = []
        for row in annotation_lists:
            row_split = row.split(' ')
            img_path = row_split[0]
            label = int(row_split[1])
            img_count = int(row_split[2])

            for idx in self._getstartindices(img_count):
                lists.append((img_path, label, idx))
        
        return lists

    def _getstartindices(self, max_range):
        return list(range(0, max_range - self.clip_dilated_frames_size + 1, self.clip_stride))

    def _getindices(self, start_id):
        return list(range(start_id, start_id + self.clip_dilated_frames_size, self.clip_dilation + 1))

    def _getdilatedsize(self):
        return int(self.clip_size * (self.clip_dilation + 1) - self.clip_dilation)

class UCFCrimeSingle(torch_data.Dataset):
    def __init__(self, root_dir, label, img_count, transforms=None, clip_size=16, clip_stride=1, clip_dilation=0):
        self.root_dir = root_dir
        self.label = label
        self.img_count = img_count
        self.transforms = transforms

        self.clip_size = clip_size
        self.clip_stride = clip_stride
        self.clip_dilation = clip_dilation
        self.clip_dilated_frames_size = self._getdilatedsize()

        self.list = self._getlist()

    def __len__(self):
        return len(self.list)

    def __getitem__(self, index):
        idx = self.list[index]
        indices = self._getindices(idx)
        img_file_template = '{:06d}.jpg'

        imgs = [Image.open(os.path.join(self.root_dir, img_file_template.format(i + 1))) for i in indices]
        if self.transforms is not None:
            imgs = self.transforms(imgs)

        return imgs

    def _getlist(self):
        return self._getstartindices(self.img_count)

    def _getstartindices(self, max_range):
        return list(range(0, max_range - self.clip_dilated_frames_size + 1, self.clip_stride))

    def _getindices(self, start_id):
        return list(range(start_id, start_id + self.clip_dilated_frames_size, self.clip_dilation + 1))

    def _getdilatedsize(self):
        return int(self.clip_size * (self.clip_dilation + 1) - self.clip_dilation)