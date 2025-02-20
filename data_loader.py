import os
import random

from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from torchvision.transforms import functional as F


class ImageFolder(data.Dataset):
    def __init__(self, root, image_size=224, mode='train', augmentation_prob=0.4):
        """Initializes image paths and preprocessing module."""
        self.root = root

        # GT : Ground Truth
        self.GT_paths = root[:-1] + '_GT/'
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.image_size = image_size
        self.mode = mode
        self.RotationDegree = [0, 90, 180, 270]
        self.augmentation_prob = augmentation_prob
        print("image count in {} path :{}".format(self.mode, len(self.image_paths)))

    def __getitem__(self, index):
        """Reads an image from a file and preprocesses it and returns."""
        image_path = self.image_paths[index]
        ground_truth_path = self.GT_paths + os.path.splitext(image_path.split('/')[-1])[0] + '_mask.png'

        image = Image.open(image_path)
        ground_truth = Image.open(ground_truth_path)

        aspect_ratio = image.size[1] / image.size[0]

        transform = []

        resize_range = random.randint(300, 320)
        transform.append(T.Resize((int(resize_range * aspect_ratio), resize_range)))
        p_transform = random.random()

        if (self.mode == 'train') and p_transform <= self.augmentation_prob:
            rotation_degree = random.randint(0, 3)
            rotation_degree = self.RotationDegree[rotation_degree]
            if (rotation_degree == 90) or (rotation_degree == 270):
                aspect_ratio = 1 / aspect_ratio

            transform.append(T.RandomRotation((rotation_degree, rotation_degree)))

            rotation_range = random.randint(-10, 10)
            transform.append(T.RandomRotation((rotation_range, rotation_range)))
            crop_range = random.randint(250, 270)
            transform.append(T.CenterCrop((int(crop_range * aspect_ratio), crop_range)))
            transform = T.Compose(transform)

            image = transform(image)
            ground_truth = transform(ground_truth)

            shift_range_left = random.randint(0, 20)
            shift_range_upper = random.randint(0, 20)
            shift_range_right = image.size[0] - random.randint(0, 20)
            shift_range_lower = image.size[1] - random.randint(0, 20)
            image = image.crop(box=(shift_range_left, shift_range_upper, shift_range_right, shift_range_lower))
            ground_truth = ground_truth.crop(
                box=(shift_range_left, shift_range_upper, shift_range_right, shift_range_lower)
            )

            if random.random() < 0.5:
                image = F.hflip(image)
                ground_truth = F.hflip(ground_truth)

            if random.random() < 0.5:
                image = F.vflip(image)
                ground_truth = F.vflip(ground_truth)

            transform = T.ColorJitter(brightness=0.2, contrast=0.2, hue=0.02)

            image = transform(image)

            transform = []

        transform.append(T.Resize((int(256 * aspect_ratio) - int(256 * aspect_ratio) % 16, 256)))
        transform.append(T.ToTensor())
        transform = T.Compose(transform)

        image = transform(image)
        ground_truth = transform(ground_truth)

        norm_ = T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        image = norm_(image)

        return image, ground_truth

    def __len__(self):
        """Returns the total number of font files."""
        return len(self.image_paths)


def get_loader(image_path, image_size, batch_size, num_workers=2, mode='train', augmentation_prob=0.4):
    """Builds and returns Dataloader."""

    dataset = ImageFolder(root=image_path, image_size=image_size, mode=mode, augmentation_prob=augmentation_prob)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  num_workers=num_workers)
    return data_loader
