#!/usr/bin/env python3

import argparse
import logging
import os

import yaml
from easydict import EasyDict
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from qartezator.trainers import make_training_model


LOGGER = logging.getLogger(__name__)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path', type=str, required=True, help='Path to training config.')
    return parser.parse_args()


def main():
    args = get_args()

    # read config
    with open(args.config_path, "r") as f:
        try:
            config = EasyDict(yaml.safe_load(f))
        except yaml.YAMLError as exc:
            print(exc)
    print(config.training_model)
    config.visualizer.outdir = os.path.join(os.getcwd(), config.visualizer.outdir)

    checkpoints_dir = os.path.join(os.getcwd(), 'models')
    os.makedirs(checkpoints_dir, exist_ok=True)

    # there is no need to suppress this logger in ddp, because it handles rank on its own
    metrics_logger = TensorBoardLogger(config.location.tb_dir, name=os.path.basename(os.getcwd()))
    metrics_logger.log_hyperparams(config)

    training_model = make_training_model(config)

    trainer_kwargs = config.trainer.kwargs

    trainer = Trainer(
        # there is no need to suppress checkpointing in ddp, because it handles rank on its own
        callbacks=ModelCheckpoint(dirpath=checkpoints_dir, **config.trainer.checkpoint_kwargs),
        logger=metrics_logger,
        default_root_dir=os.getcwd(),
        **trainer_kwargs
    )
    trainer.fit(training_model)


if __name__ == '__main__':
    main()
