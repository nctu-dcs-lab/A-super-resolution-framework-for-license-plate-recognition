import torch
from torch.nn import functional as F

from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.models.sr_model import SRModel
from swinfir.models.model_util import mixup

from collections import OrderedDict

import torch
import torch.nn as nn


@MODEL_REGISTRY.register()
class SwinFIRModel_2(SRModel):

    def feed_data(self, data, phase='val'):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)
        if 'phase' in data and 'use_mixup' in data:
            if data['phase'] == 'train' and data['use_mixup']:
                self.lq, self.gt = mixup(self.lq, self.gt)

    def test(self):
        # pad to multiplication of window_size
        window_size = self.opt['network_g']['window_size']
        scale = self.opt.get('scale', 1)
        _, _, h, w = self.lq.size()
        mod_pad_h = (h // window_size + 1) * window_size - h
        mod_pad_w = (w // window_size + 1) * window_size - w
        img = torch.cat([self.lq, torch.flip(self.lq, [2])], 2)[:, :, :h + mod_pad_h, :]
        img = torch.cat([img, torch.flip(img, [3])], 3)[:, :, :, :w + mod_pad_w]

        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(img)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(img)
            self.net_g.train()

        self.output = self.output[..., :h * scale, :w * scale]

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # # define losses
        # if train_opt.get('pixel_opt'):
        #     self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        # else:
        #     self.cri_pix = None

        # if train_opt.get('perceptual_opt'):
        #     self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        # else:
        #     self.cri_perceptual = None

        # if self.cri_pix is None and self.cri_perceptual is None:
        #     raise ValueError('Both pixel and perceptual losses are None.')

        # define losses
        self.loss_options = {}

        for key in ['pixel_opt', 'pixel_opt_2', 'perceptual_opt','perceptual_opt_2']:
            if key in train_opt:
                self.loss_options[key] = build_loss(train_opt[key]).to(self.device)

        if not self.loss_options:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        # Define gradient clipping value
        clip_value = 0.5

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        # if self.cri_pix:
        #     l_pix = self.cri_pix(self.output, self.gt)
        #     l_total += l_pix
        #     loss_dict['l_pix'] = l_pix
        # # perceptual loss
        # if self.cri_perceptual:
        #     l_percep, l_style = self.cri_perceptual(self.output, self.gt)
        #     if l_percep is not None:
        #         l_total += l_percep
        #         loss_dict['l_percep'] = l_percep
        #     if l_style is not None:
        #         l_total += l_style
        #         loss_dict['l_style'] = l_style

        for key, criterion in self.loss_options.items():
            if key.startswith('pixel'):
                loss = criterion(self.output, self.gt)
                loss_dict[key + '_l'] = loss
                l_total += loss
            elif key.startswith('perceptual'):
                loss_percep, loss_style = criterion(self.output, self.gt)
                if loss_percep is not None:
                    loss_dict[key + '_percep_l'] = loss_percep
                    l_total += loss_percep
                if loss_style is not None:
                    loss_dict[key + '_style_l'] = loss_style
                    l_total += loss_style
            

        l_total.backward()

        # Clip gradients
        # nn.utils.clip_grad_norm_(self.net_g.parameters(), clip_value)

        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)
