from pathlib import Path
from typing import Union, List, Optional

import pytorch_lightning as pl
from torch.utils.data import DataLoader

from qartezator.data.dataset import QartezatorDataset
from qartezator.data.datautils import seed_worker
from qartezator.data.transforms import get_transforms, get_common_augmentations
from qartezator.data.typing import TransformType


class QartezatorDataModule(pl.LightningDataModule):

    def __init__(
        self,
        root_path: Union[Path, str],
        train_txt_path: Union[Path, str],
        val_txt_path: Union[Path, str],
        test_txt_path: Union[Path, str],
        train_batch_size: int = 32,
        val_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 4,
        input_size: int = 256,
        pad_to_modulo: int = 32,
        common_transform: Optional[TransformType] = None,
        source_transform: Optional[TransformType] = None,
        mean: List[float] = None,
        std: List[float] = None
    ) -> None:
        super().__init__()

        self.root_path = root_path
        self.train_txt_path = train_txt_path
        self.val_txt_path = val_txt_path
        self.test_txt_path = test_txt_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.input_size = input_size
        self.pad_to_modulo = pad_to_modulo
        self.common_transform = common_transform
        self.source_transform = source_transform
        self.mean = mean
        self.std = std

    def train_dataloader(self) -> DataLoader:
        # source_augmentations = ...

        common_transform = self.common_transform
        if self.common_transform is None:
            common_augmentations = get_common_augmentations(self.input_size)
            common_transform = get_transforms(mean=self.mean, std=self.std, augmentations=common_augmentations)

        dataset = QartezatorDataset(
            root_path=self.root_path,
            split_file_path=self.train_txt_path,
            source_transform=self.source_transform,
            common_transform=common_transform,
            pad_to_modulo=self.pad_to_modulo
        )

        return DataLoader(
            dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
            worker_init_fn=seed_worker
        )

    def val_dataloader(self) -> DataLoader:
        transform = get_transforms(mean=self.mean, std=self.std)

        dataset = QartezatorDataset(
            root_path=self.root_path,
            split_file_path=self.val_txt_path,
            common_transform=transform,
            pad_to_modulo=self.pad_to_modulo
        )

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        transform = get_transforms(mean=self.mean, std=self.std)

        dataset = QartezatorDataset(
            root_path=self.root_path,
            split_file_path=self.test_txt_path,
            common_transform=transform,
            pad_to_modulo=self.pad_to_modulo
        )

        return DataLoader(
            dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )


if __name__ == '__main__':
    root_path = 'data/maps'
    train_txt_path = 'assets/train.txt'
    val_txt_path = 'assets/val.txt'
    test_txt_path = 'assets/test.txt'
    dm = QartezatorDataModule(
        root_path=root_path,
        train_txt_path=train_txt_path,
        val_txt_path=val_txt_path,
        test_txt_path=test_txt_path,
        input_size=256
    )
    train_dataloader = dm.train_dataloader()
    for batch in train_dataloader:
        source, target = batch
        print(f'Source batch dtype: {source.dtype}')
        print(f'Target batch dtype: {target.dtype}\n')
        print(f'Source batch shape: {source.shape}')
        print(f'Target batch shape: {target.shape}\n')
        break
