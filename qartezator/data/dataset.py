from pathlib import Path
from typing import Union, Tuple, Optional

from torch.utils.data import Dataset

from qartezator.data.datautils import load_image, pad_img_to_modulo
from qartezator.data.typing import TransformType, DatasetElement


class QartezatorDataset(Dataset):
    def __init__(
        self,
        root_path: Union[str, Path],
        split_file_path: Union[str, Path],
        source_transform: Optional[TransformType] = None,
        common_transform: Optional[TransformType] = None,
        pad_to_modulo: int = 32
    ):
        self.root_path = Path(root_path)
        self.split_file_path = split_file_path
        self.source_transform = source_transform
        self.common_transform = common_transform
        self.pad_to_modulo = pad_to_modulo
        with open(split_file_path) as f:
            self.img_paths = f.read().splitlines()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index: int) -> Tuple[DatasetElement, DatasetElement]:
        img_path = self.root_path / self.img_paths[index]
        img = load_image(img_path)
        img_h, img_w = img.shape[:2]
        source_img = img[:, :img_w//2]
        target_img = img[:, img_w//2:]
        if self.pad_to_modulo > 0:
            source_img = pad_img_to_modulo(source_img, self.pad_to_modulo)
            target_img = pad_img_to_modulo(target_img, self.pad_to_modulo)
        if self.source_transform is not None:
            transformed = self.source_transform(image=source_img)
            source_img = transformed['image']
        if self.common_transform is not None:
            transformed = self.common_transform(image=source_img, target=target_img)
            source_img = transformed['image']
            target_img = transformed['target']
        source_img = source_img / 255.0
        target_img = target_img / 255.0
        return source_img, target_img


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from qartezator.data.transforms import get_common_augmentations, get_source_augmentations

    # Create QartezatorDataset and take the first sample
    ds = QartezatorDataset(
        root_path='./data/maps',
        split_file_path='./assets/train.txt',
        common_transform=get_common_augmentations(600),
        source_transform=get_source_augmentations()
    )
    sample_source_img, sample_target_img = ds[10]

    # Print types of source and target images
    print(f'Type of the source image: {type(sample_source_img)}')
    print(f'Type of the target image: {type(sample_target_img)}\n')

    # Print shapes of source and target images
    print(f'Shape of the source image: {sample_source_img.shape}')
    print(f'Shape of the target image: {sample_target_img.shape}\n')

    # Plot sample source image
    plt.subplot(121)
    plt.imshow(sample_source_img)
    plt.title('Source image'), plt.xticks([]), plt.yticks([])

    # Plot sample target image
    plt.subplot(122)
    plt.imshow(sample_target_img)
    plt.title('Target image'), plt.xticks([]), plt.yticks([])

    # Show the plot
    plt.show()
