from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
import os
import csv
import torch
from torchvision.transforms import Normalize
import numpy as np

# Labels
#
# modern infrastructure: inf:5
#
# crowd: cro:4
#
# human:hum:3
#
# animal:ani:2
#
# nature:nat:1

label_dict = {0: "nature",
              1: "animal",
              2: "human",
              3: "crowd",
              4: "infra"
              }


class Thermal(Dataset):
    def __init__(self, thermal_dir, color_dir, annot_path, split='train', transform=None, normalize=True, label_filter = [1,2,3,4,5], cut=None, ds_reduction=None, class_limit=100):
        self.label_filter = label_filter
        self.filter_map = {k:v for (v,k) in enumerate(self.label_filter)}
        self.transform = transform
        self.annot_path = annot_path
        self.split = split
        self.thermal_dir = thermal_dir
        self.color_dir = color_dir
        self.normalize = normalize
        self.thermal_list = list()
        self.norm_thermal_list = list()
        self.color_list = list()
        self.norm_color_list = list()
        self.color_suffix = '_color'
        self.thermal_files = list()
        self.thermal_ext = 'jpg'
        self.color_ext = 'png'
        self.boxes = dict()
        self.labels = dict()
        self.reduction = ds_reduction
        self.load_thermals()
        self.load_colors()
        self.class_counter = dict.fromkeys(range(6), 0)
        self.class_count_limit = class_limit
        self.load_annotations()
        self.color_normalizer = None
        self.thermal_normalizer = None
        self.original_labels = dict()

        if cut is None:
            self.cut = int(0.8 * len(self.color_list))
        else:
            self.cut = int(cut * len(self.color_list))
        if self.split == 'train':
            self.thermal_list = self.thermal_list[:self.cut]
            self.color_list = self.color_list[:self.cut]
        elif self.split == 'validate':
            self.thermal_list = self.thermal_list[self.cut:]
            self.color_list = self.color_list[self.cut:]
        if self.normalize:
            print('Normalizing images ')
            self.normalize_images()

        print('## Loaded {} dataset with {} samples ##'.format(self.split, len(self.thermal_list)))


    def load_annotations(self):
        annots = None
        with open(self.annot_path, newline='') as csvfile:
            annots = csv.reader(csvfile, delimiter=',', quotechar='|')
            annots = list(annots)
        frames = [annot[0] for annot in annots]
        frames = set(frames)
        self.boxes = dict.fromkeys(frames, 0)
        self.labels = dict.fromkeys(frames, 0)
        self.original_labels = dict.fromkeys(frames, 0)
        frames_to_remove = list()
        remove_indices = list()
        for annot in annots[1:]:
            if int(annot[-1]) in self.label_filter:
                if self.boxes[annot[0]] == 0:
                    self.boxes[annot[0]] = list()
                box = [int(dim) for dim in annot[1:-1]]
                if box[2] == 0 or box[3] == 0:
                    print('###Warning####', box, annot[0])
                box[2] = box[0] + box[2]
                box[3] = box[1] + box[3]
                self.boxes[annot[0]].append(box)
                if self.labels[annot[0]] == 0:
                    self.labels[annot[0]] = list()
                    self.original_labels[annot[0]] = list()
                self.labels[annot[0]].append(self.filter_map[int(annot[-1])])#Put into range from 0 to 4
                self.original_labels[annot[0]].append(int(annot[-1]))


        # print('#DEBUG', self.boxes)
        for i, file in enumerate(self.thermal_files):
            if self.boxes[file[:-4]] == 0 or len(self.boxes[file[:-4]]) > 1:
                remove_indices.append(i)

        self.color_list = [element for i, element in enumerate(self.color_list) if i not in remove_indices]
        self.thermal_list = [element for i, element in enumerate(self.thermal_list) if i not in remove_indices]
        self.thermal_files = [element for i, element in enumerate(self.thermal_files) if i not in remove_indices]




        #workaround to balance instance number after the pre processing
        remove_indices = list()
        for i, file in enumerate(self.thermal_files):
            self.class_counter[self.original_labels[file[:-4]][0]] += 1
            if self.class_counter[self.original_labels[file[:-4]][0]] > self.class_count_limit:
                remove_indices.append(i)



        self.color_list = [element for i, element in enumerate(self.color_list) if i not in remove_indices]
        self.thermal_list = [element for i, element in enumerate(self.thermal_list) if i not in remove_indices]
        self.thermal_files = [element for i, element in enumerate(self.thermal_files) if i not in remove_indices]
        print(len(self.thermal_files))
        print('class instances processed: ', self.class_counter)
        # print('DEBUG size ', len(self.thermal_files))
        # for frame in self.thermal_files:
        #     print('DEBUG', frame[:-4], self.labels[frame[:-4]], self.original_labels[frame[:-4]])


    def load_thermals(self):
        #Sort files alphabetical order
        self.thermal_files = os.listdir(self.thermal_dir)
        self.thermal_files.sort()
        if self.reduction is not None:
            self.thermal_files = self.thermal_files[:self.reduction]
        # print('debug :',self.reduction, len(self.thermal_files))
        for thermal_file in self.thermal_files:
            if thermal_file[-3:] == self.thermal_ext:
                thermal_t = read_image(os.path.join(self.thermal_dir,thermal_file))
                self.thermal_list.append(thermal_t[0, :, :].unsqueeze(0).float()) #Use only one channel
        # print('debug :', len(self.thermal_list))

    def load_colors(self):
        # Sort files alphabetical order
        for thermal_file in self.thermal_files:
            color_equiv = thermal_file[:-4] + self.color_suffix + '.' + self.color_ext
            color_path = os.path.join(self.color_dir, color_equiv)
            if os.path.isfile(color_path):
                color_t = read_image(color_path).float()
                self.color_list.append(color_t)

    def normalize_images(self):
        # for color_im in self.color_list:
        #     norm_im = color_im/255
        #     self.norm_color_list.append(norm_im)
        #
        # for thermal_im in self.thermal_list:
        #     norm_im = thermal_im/255
        #     self.norm_thermal_list.append(norm_im)

        #print('Debug',self.color_list[0].size(), [torch.mean(color_im,(0,1)) for color_im in self.color_list])
        mean_color = np.mean([torch.mean(color_im,(1,2)).numpy() for color_im in self.color_list], axis=0)
        std_color = np.mean([torch.std(color_im,(1,2)).numpy() for color_im in self.color_list],axis=0)
        mean_color = mean_color / 255
        std_color = std_color / 255
        print(f'Color normalization: mean {mean_color} std {std_color}')
        self.color_normalizer = Normalize(mean_color, std_color)
        for color_im in self.color_list:
            norm_im = self.color_normalizer(color_im/255)
            self.norm_color_list.append(norm_im)


        mean_thermal = np.mean([torch.mean(thermal_im,1).numpy() for thermal_im in self.thermal_list])
        std_thermal = np.mean([torch.mean(thermal_im,1).numpy() for thermal_im in self.thermal_list])
        mean_thermal = mean_thermal / 255
        std_thermal = std_thermal / 255
        print(f'Thermal normalization: mean {mean_thermal} std {std_thermal}')
        self.thermal_normalizer = Normalize(mean_thermal,
                                            std_thermal)
        for thermal_im in self.thermal_list:
            norm_im = self.thermal_normalizer(thermal_im/255)
            self.norm_thermal_list.append(norm_im)

    def __len__(self):
        return len(self.color_list)

    def __repr__(self):
        return "Total image pairs: {}".format(len(self.thermal_list))

    def __getitem__(self, index: int):
        if self.normalize:
            color = self.norm_color_list[index]
            thermal = self.norm_thermal_list[index]
        else:
            color = self.color_list[index]
            thermal = self.thermal_list[index]
        frame = self.thermal_files[index][:-4]

        boxes = torch.FloatTensor(self.boxes[frame])
        labels = torch.LongTensor(self.labels[frame])
        return color, thermal, boxes, labels


def collate_fn(batch):
    """
    Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).
    This describes how to combine these tensors of different sizes. We use lists.
    Note: this need not be defined in this Class, can be standalone.
    :param batch: an iterable of N sets from __getitem__()
    :return: a tensor of images, lists of varying-size tensors of bounding boxes, labels
    """

    colors = list()
    thermals = list()
    boxes = list()
    labels = list()

    for b in batch:
        colors.append(b[0])
        thermals.append(b[1])
        boxes.append(b[2])
        labels.append(b[3])

    colors = torch.stack(colors, dim=0)
    thermals = torch.stack(thermals, dim=0)

    return colors, thermals, boxes, labels  # tensor (N, 3, 300, 300),tensor (N, 1, 300, 300), 2 lists of N tensors each

if __name__ == '__main__':
    thermal_ds = Thermal()