import copy
import logging
from typing import Dict, Tuple

import pandas as pd
import pytorch_lightning as ptl
import torch
import torch.nn as nn

from qartezator.evaluation import make_evaluator
from qartezator.data.datamodule import QartezatorDataModule
from qartezator.data.transforms import get_source_augmentations
from qartezator.losses.adversarial import make_discrim_loss
from qartezator.losses.perceptual import PerceptualLoss, ResNetPL
from qartezator.models.generators import FFCResNetGenerator
from qartezator.models.discriminators import NLayerDiscriminator
from qartezator.visualizers import make_visualizer
from qartezator.utils.lamautils import add_prefix_to_keys, average_dicts, set_requires_grad, flatten_dict, \
    get_has_ddp_rank

LOGGER = logging.getLogger(__name__)


def make_optimizer(parameters, kind='adamw', **kwargs):
    if kind == 'adam':
        optimizer_class = torch.optim.Adam
    elif kind == 'adamw':
        optimizer_class = torch.optim.AdamW
    else:
        raise ValueError(f'Unknown optimizer kind {kind}')
    return optimizer_class(parameters, **kwargs)


def update_running_average(result: nn.Module, new_iterate_model: nn.Module, decay=0.999):
    with torch.no_grad():
        res_params = dict(result.named_parameters())
        new_params = dict(new_iterate_model.named_parameters())

        for k in res_params.keys():
            res_params[k].data.mul_(decay).add_(new_params[k].data, alpha=1 - decay)


class BaseTrainingModule(ptl.LightningModule):
    def __init__(self, config, use_ddp, *args, predict_only=False, visualize_each_iters=100,
                 average_generator=False, generator_avg_beta=0.999, average_generator_start_step=30000,
                 average_generator_period=10, **kwargs):
        super().__init__(*args, **kwargs)
        LOGGER.info('BaseTrainingModule init called')

        self.config = config

        self.datamodule = QartezatorDataModule(**self.config.datamodule,
                                               source_transform=get_source_augmentations())
        self.generator = FFCResNetGenerator(**self.config.generator)
        self.use_ddp = use_ddp

        if not get_has_ddp_rank():
            LOGGER.info(f'Generator\n{self.generator}')

        if not predict_only:
            self.save_hyperparameters(self.config)
            self.discriminator = NLayerDiscriminator(**self.config.discriminator)
            self.adversarial_loss = make_discrim_loss(**self.config.losses.adversarial)
            self.visualizer = make_visualizer(**self.config.visualizer)
            self.val_evaluator = make_evaluator(**self.config.evaluator)
            self.test_evaluator = make_evaluator(**self.config.evaluator)

            if not get_has_ddp_rank():
                LOGGER.info(f'Discriminator\n{self.discriminator}')

            self.average_generator = average_generator
            self.generator_avg_beta = generator_avg_beta
            self.average_generator_start_step = average_generator_start_step
            self.average_generator_period = average_generator_period
            self.generator_average = None
            self.last_generator_averaging_step = -1

            if self.config.losses.get("l1", {"weight_known": 0})['weight_known'] > 0:
                self.loss_l1 = nn.L1Loss(reduction='none')

            if self.config.losses.get("mse", {"weight": 0})['weight'] > 0:
                self.loss_mse = nn.MSELoss(reduction='none')

            if self.config.losses.perceptual.weight > 0:
                self.loss_pl = PerceptualLoss()

            if self.config.losses.get("resnet_pl", {"weight": 0})['weight'] > 0:
                self.loss_resnet_pl = ResNetPL(**self.config.losses.resnet_pl)
            else:
                self.loss_resnet_pl = None

        self.visualize_each_iters = visualize_each_iters
        LOGGER.info('BaseTrainingModule init done')

    def configure_optimizers(self):
        discriminator_params = list(self.discriminator.parameters())
        return [
            dict(optimizer=make_optimizer(self.generator.parameters(), **self.config.optimizers.generator)),
            dict(optimizer=make_optimizer(discriminator_params, **self.config.optimizers.discriminator)),
        ]

    def train_dataloader(self):
        return self.datamodule.train_dataloader()

    def val_dataloader(self):
        return [self.datamodule.val_dataloader(), self.datamodule.val_dataloader()]

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        self._is_training_step = True
        return self._do_step(batch, batch_idx, mode='train', optimizer_idx=optimizer_idx)

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx == 0:
            mode = 'val'
        elif dataloader_idx == 1:
            mode = 'test'
        self._is_training_step = False
        return self._do_step(batch, batch_idx, mode=mode)

    def training_step_end(self, batch_parts_outputs):
        if self.training and self.average_generator \
                and self.global_step >= self.average_generator_start_step \
                and self.global_step >= self.last_generator_averaging_step + self.average_generator_period:
            if self.generator_average is None:
                self.generator_average = copy.deepcopy(self.generator)
            else:
                update_running_average(self.generator_average, self.generator, decay=self.generator_avg_beta)
            self.last_generator_averaging_step = self.global_step

        full_loss = (batch_parts_outputs['loss'].mean()
                     if torch.is_tensor(batch_parts_outputs['loss'])  # loss is not tensor when no discriminator used
                     else torch.tensor(batch_parts_outputs['loss']).float().requires_grad_(True))
        log_info = {k: v.mean() for k, v in batch_parts_outputs['log_info'].items()}
        self.log_dict(log_info, on_step=True, on_epoch=False)
        return full_loss

    def validation_epoch_end(self, outputs):
        outputs = [step_out for out_group in outputs for step_out in out_group]
        averaged_logs = average_dicts(step_out['log_info'] for step_out in outputs)
        self.log_dict({k: v.mean() for k, v in averaged_logs.items()})

        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)

        # standard validation
        val_evaluator_states = [s['val_evaluator_state'] for s in outputs if 'val_evaluator_state' in s]
        val_evaluator_res = self.val_evaluator.evaluation_end(states=val_evaluator_states)
        val_evaluator_res_df = pd.DataFrame(val_evaluator_res).stack(1).unstack(0)
        val_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
        LOGGER.info(f'Validation metrics after epoch #{self.current_epoch}, '
                    f'total {self.global_step} iterations:\n{val_evaluator_res_df}')

        for k, v in flatten_dict(val_evaluator_res).items():
            self.log(f'val_{k}', v)

        # standard visual test
        test_evaluator_states = [s['test_evaluator_state'] for s in outputs
                                 if 'test_evaluator_state' in s]
        test_evaluator_res = self.test_evaluator.evaluation_end(states=test_evaluator_states)
        test_evaluator_res_df = pd.DataFrame(test_evaluator_res).stack(1).unstack(0)
        test_evaluator_res_df.dropna(axis=1, how='all', inplace=True)
        LOGGER.info(f'Test metrics after epoch #{self.current_epoch}, '
                    f'total {self.global_step} iterations:\n{test_evaluator_res_df}')

        for k, v in flatten_dict(test_evaluator_res).items():
            self.log(f'test_{k}', v)

    def _do_step(self, batch, batch_idx, mode='train', optimizer_idx=None):
        source_img, target_img = batch

        if optimizer_idx == 0:  # step for generator
            set_requires_grad(self.generator, True)
            set_requires_grad(self.discriminator, False)
        elif optimizer_idx == 1:  # step for discriminator
            set_requires_grad(self.generator, False)
            set_requires_grad(self.discriminator, True)

        predicted_img = self(source_img)

        total_loss = 0
        metrics = {}

        if optimizer_idx is None or optimizer_idx == 0:  # step for generator
            total_loss, metrics = self.generator_loss(source_img, target_img, predicted_img)

        elif optimizer_idx is None or optimizer_idx == 1:  # step for discriminator
            if self.config.losses.adversarial.weight > 0:
                total_loss, metrics = self.discriminator_loss(source_img, target_img, predicted_img)

        batch_dict = {
            'source_img': source_img,
            'predicted_img': predicted_img,
            'target_img': target_img
        }

        if self.get_ddp_rank() in (None, 0) and (batch_idx % self.visualize_each_iters == 0 or mode == 'test'):
            vis_suffix = f'_{mode}'
            self.visualizer(self.current_epoch, batch_idx, batch_dict, suffix=vis_suffix)

        metrics_prefix = f'{mode}_'
        result = dict(loss=total_loss, log_info=add_prefix_to_keys(metrics, metrics_prefix))
        if mode == 'val':
            result['val_evaluator_state'] = self.val_evaluator.process_batch(batch_dict)
        elif mode == 'test':
            result['test_evaluator_state'] = self.test_evaluator.process_batch(batch_dict)

        return result

    def get_current_generator(self, no_average=False):
        if not no_average and not self.training and self.average_generator and self.generator_average is not None:
            return self.generator_average
        return self.generator

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        raise NotImplementedError()

    def generator_loss(self, source_img, target_img, predicted_img) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def discriminator_loss(self, source_img, target_img, predicted_img) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        raise NotImplementedError()

    def get_ddp_rank(self):
        return None
