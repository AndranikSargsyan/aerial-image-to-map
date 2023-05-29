import numpy as np
from PIL import Image


root_path =  './data/maps'
train_txt_path = './qartezator/assets/train.txt'
val_txt_path = './qartezator/assets/val.txt'
test_txt_path = './qartezator/assets/test.txt'

ds = QartezatorDataset(
    root_path=root_path,
    split_file_path=train_txt_path,
    common_transform=get_common_augmentations(256)
)
sample_source_img, sample_target_img = ds[42]

Image.fromarray((sample_source_img*255).astype(np.uint8))

Image.fromarray((sample_target_img*255).astype(np.uint8))

dm = QartezatorDataModule(
    root_path=root_path,
    train_txt_path=train_txt_path,
    val_txt_path=val_txt_path,
    test_txt_path=test_txt_path,
    input_size=256
)
train_dataloader = dm.train_dataloader()
val_dataloader = dm.val_dataloader()
test_dataloader = dm.test_dataloader()

for batch in train_dataloader:
    source, target = batch
    print(f'Source batch shape: {source.shape}')
    print(f'Target batch shape: {target.shape}\n')
    break

len(train_dataloader.dataset)
len(val_dataloader.dataset)
len(test_dataloader.dataset)