import logging

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from qartezator.losses.feature_matching import feature_matching_loss, l1_loss
from qartezator.trainers.base import BaseTrainingModule
from qartezator.utils.lamautils import add_prefix_to_keys

LOGGER = logging.getLogger(__name__)


class DefaultTrainingModule(BaseTrainingModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, source_img):
        predicted_img = self.generator(source_img)
        return predicted_img

    def generator_loss(self, source_img, target_img, predicted_img):
        # L1
        l1_value = l1_loss(predicted_img, target_img)

        total_loss = l1_value
        metrics = dict(gen_l1=l1_value)

        # vgg-based perceptual loss
        if self.config.losses.perceptual.weight > 0:
            pl_value = self.loss_pl(predicted_img, target_img).sum() * self.config.losses.perceptual.weight
            total_loss = total_loss + pl_value
            metrics['gen_pl'] = pl_value

        # discriminator
        # adversarial_loss calls backward by itself
        self.adversarial_loss.pre_generator_step(real_batch=target_img, fake_batch=predicted_img,
                                                 generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(torch.cat([source_img, target_img], dim=1))
        discr_fake_pred, discr_fake_features = self.discriminator(torch.cat([source_img, predicted_img], dim=1))
        adv_gen_loss, adv_metrics = self.adversarial_loss.generator_loss(real_batch=target_img,
                                                                         fake_batch=predicted_img,
                                                                         discr_real_pred=discr_real_pred,
                                                                         discr_fake_pred=discr_fake_pred)
        total_loss = total_loss + adv_gen_loss
        metrics['gen_adv'] = adv_gen_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        # feature matching
        if self.config.losses.feature_matching.weight > 0:
            fm_value = feature_matching_loss(discr_fake_features, discr_real_features) * self.config.losses.feature_matching.weight
            total_loss = total_loss + fm_value
            metrics['gen_fm'] = fm_value

        if self.loss_resnet_pl is not None:
            resnet_pl_value = self.loss_resnet_pl(predicted_img, target_img)
            total_loss = total_loss + resnet_pl_value
            metrics['gen_resnet_pl'] = resnet_pl_value

        return total_loss, metrics

    def discriminator_loss(self, source_img, target_img, predicted_img):
        total_loss = 0
        metrics = {}

        self.adversarial_loss.pre_discriminator_step(real_batch=target_img, fake_batch=predicted_img,
                                                     generator=self.generator, discriminator=self.discriminator)
        discr_real_pred, discr_real_features = self.discriminator(torch.cat([source_img, target_img], dim=1))
        discr_fake_pred, discr_fake_features = self.discriminator(torch.cat([source_img, predicted_img], dim=1))
        adv_discr_loss, adv_metrics = self.adversarial_loss.discriminator_loss(
            real_batch=target_img,
            fake_batch=predicted_img,
            discr_real_pred=discr_real_pred,
            discr_fake_pred=discr_fake_pred,
        )
        total_loss = total_loss + adv_discr_loss
        metrics['discr_adv'] = adv_discr_loss
        metrics.update(add_prefix_to_keys(adv_metrics, 'adv_'))

        return total_loss, metrics
