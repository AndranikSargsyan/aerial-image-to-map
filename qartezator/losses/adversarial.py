from typing import Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# CycleGAN
class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss


def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    """Calculate the gradient penalty loss, used in WGAN-GP paper https://arxiv.org/abs/1704.00028

    Arguments:
        netD (network)              -- discriminator network
        real_data (tensor array)    -- real images
        fake_data (tensor array)    -- generated images from the generator
        device (str)                -- GPU / CPU: from torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        type (str)                  -- if we mix real and fake data or not [real | fake | mixed].
        constant (float)            -- the constant used in formula ( ||gradient||_2 - constant)^2
        lambda_gp (float)           -- weight for this loss

    Returns the gradient penalty loss
    """
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None


# LaMa
class BaseAdversarialLoss:
    def pre_generator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                           generator: nn.Module, discriminator: nn.Module):
        """
        Prepare for generator step
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def pre_discriminator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                               generator: nn.Module, discriminator: nn.Module):
        """
        Prepare for discriminator step
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param generator:
        :param discriminator:
        :return: None
        """

    def generator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                       discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate generator loss
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for real_batch
        :param discr_fake_pred: Tensor, discriminator output for fake_batch
        :return: total generator loss along with some values that might be interesting to log
        """
        raise NotImplemented()

    def discriminator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                           discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Calculate discriminator loss and call .backward() on it
        :param real_batch: Tensor, a batch of real samples
        :param fake_batch: Tensor, a batch of samples produced by generator
        :param discr_real_pred: Tensor, discriminator output for real_batch
        :param discr_fake_pred: Tensor, discriminator output for fake_batch
        :return: total discriminator loss along with some values that might be interesting to log
        """
        raise NotImplemented()


def make_r1_gp(discr_real_pred, real_batch):
    if torch.is_grad_enabled():
        grad_real = torch.autograd.grad(outputs=discr_real_pred.sum(), inputs=real_batch, create_graph=True)[0]
        grad_penalty = (grad_real.view(grad_real.shape[0], -1).norm(2, dim=1) ** 2).mean()
    else:
        grad_penalty = 0
    real_batch.requires_grad = False

    return grad_penalty


class NonSaturatingWithR1(BaseAdversarialLoss):
    def __init__(self, gp_coef=5, weight=1):
        self.gp_coef = gp_coef
        self.weight = weight

    def generator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                       discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        fake_loss = F.softplus(-discr_fake_pred)
        return fake_loss.mean() * self.weight, dict()

    def pre_discriminator_step(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                               generator: nn.Module, discriminator: nn.Module):
        real_batch.requires_grad = True

    def discriminator_loss(self, real_batch: torch.Tensor, fake_batch: torch.Tensor,
                           discr_real_pred: torch.Tensor, discr_fake_pred: torch.Tensor) \
            -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        real_loss = F.softplus(-discr_real_pred)
        grad_penalty = make_r1_gp(discr_real_pred, real_batch) * self.gp_coef
        fake_loss = F.softplus(discr_fake_pred)

        sum_discr_loss = real_loss + grad_penalty + fake_loss
        metrics = dict(discr_real_out=discr_real_pred.mean(),
                       discr_fake_out=discr_fake_pred.mean(),
                       discr_real_gp=grad_penalty)
        return sum_discr_loss.mean(), metrics


def make_discrim_loss(kind, **kwargs):
    if kind == 'r1':
        return NonSaturatingWithR1(**kwargs)
    raise ValueError(f'Unknown adversarial loss kind {kind}')

