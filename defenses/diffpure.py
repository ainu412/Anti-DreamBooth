# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import logging
import yaml
import os
import time
from PIL import Image
import glob
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from diffpure_package.runners.diffpure_ddpm import Diffusion
from diffpure_package.runners.diffpure_guided import GuidedDiffusion
from diffpure_package.runners.diffpure_sde import RevGuidedDiffusion

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import sys
import argparse
from typing import Any

import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from robustbench import load_model
import diffpure_package.data
import torch
import torch.nn.functional as F

criterion = torch.nn.CrossEntropyLoss()

class BPDA_EOT_Attack():
    def __init__(self, model, adv_eps=8.0/255, eot_defense_reps=150, eot_attack_reps=15):
        self.model = model

        self.config = {
            'eot_defense_ave': 'logits',
            'eot_attack_ave': 'logits',
            'eot_defense_reps': eot_defense_reps,
            'eot_attack_reps': eot_attack_reps,
            'adv_steps': 50,
            'adv_norm': 'l_inf',
            'adv_eps': adv_eps,
            'adv_eta': 2.0 / 255,
            'log_freq': 10
        }

        print(f'BPDA_EOT config: {self.config}')

    def purify(self, x):
        return self.model(x, mode='purify')

    def eot_defense_prediction(seslf, logits, reps=1, eot_defense_ave=None):
        if eot_defense_ave == 'logits':
            logits_pred = logits.view([reps, int(logits.shape[0]/reps), logits.shape[1]]).mean(0)
        elif eot_defense_ave == 'softmax':
            logits_pred = F.softmax(logits, dim=1).view([reps, int(logits.shape[0]/reps), logits.shape[1]]).mean(0)
        elif eot_defense_ave == 'logsoftmax':
            logits_pred = F.log_softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
        elif reps == 1:
            logits_pred = logits
        else:
            raise RuntimeError('Invalid ave_method_pred (use "logits" or "softmax" or "logsoftmax")')
        _, y_pred = torch.max(logits_pred, 1)
        return y_pred

    def eot_attack_loss(self, logits, y, reps=1, eot_attack_ave='loss'):
        if eot_attack_ave == 'logits':
            logits_loss = logits.view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
            y_loss = y
        elif eot_attack_ave == 'softmax':
            logits_loss = torch.log(F.softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0))
            y_loss = y
        elif eot_attack_ave == 'logsoftmax':
            logits_loss = F.log_softmax(logits, dim=1).view([reps, int(logits.shape[0] / reps), logits.shape[1]]).mean(0)
            y_loss = y
        elif eot_attack_ave == 'loss':
            logits_loss = logits
            y_loss = y.repeat(reps)
        else:
            raise RuntimeError('Invalid ave_method_eot ("logits", "softmax", "logsoftmax", "loss")')
        loss = criterion(logits_loss, y_loss)
        return loss

    def predict(self, X, y, requires_grad=True, reps=1, eot_defense_ave=None, eot_attack_ave='loss'):
        if requires_grad:
            logits = self.model(X, mode='classify')
        else:
            with torch.no_grad():
                logits = self.model(X.data, mode='classify')

        y_pred = self.eot_defense_prediction(logits.detach(), reps, eot_defense_ave)
        correct = torch.eq(y_pred, y)
        loss = self.eot_attack_loss(logits, y, reps, eot_attack_ave)

        return correct.detach(), loss

    def pgd_update(self, X_adv, grad, X, adv_norm, adv_eps, adv_eta, eps=1e-10):
        if adv_norm == 'l_inf':
            X_adv.data += adv_eta * torch.sign(grad)
            X_adv = torch.clamp(torch.min(X + adv_eps, torch.max(X - adv_eps, X_adv)), min=0, max=1)
        elif adv_norm == 'l_2':
            X_adv.data += adv_eta * grad / grad.view(X.shape[0], -1).norm(p=2, dim=1).view(X.shape[0], 1, 1, 1)
            dists = (X_adv - X).view(X.shape[0], -1).norm(dim=1, p=2).view(X.shape[0], 1, 1, 1)
            X_adv = torch.clamp(X + torch.min(dists, adv_eps*torch.ones_like(dists))*(X_adv-X)/(dists+eps), min=0, max=1)
        else:
            raise RuntimeError('Invalid adv_norm ("l_inf" or "l_2"')
        return X_adv

    def purify_and_predict(self, X, y, purify_reps=1, requires_grad=True):
        X_repeat = X.repeat([purify_reps, 1, 1, 1])
        X_repeat_purified = self.purify(X_repeat).detach().clone()
        X_repeat_purified.requires_grad_()
        correct, loss = self.predict(X_repeat_purified, y, requires_grad, purify_reps,
                                     self.config['eot_defense_ave'], self.config['eot_attack_ave'])
        if requires_grad:
            X_grads = torch.autograd.grad(loss, [X_repeat_purified])[0]
            # average gradients over parallel samples for EOT attack
            attack_grad = X_grads.view([purify_reps]+list(X.shape)).mean(dim=0)
            return correct, attack_grad
        else:
            return correct, None

    def eot_defense_verification(self, X_adv, y, correct, defended):
        for verify_ind in range(correct.nelement()):
            if correct[verify_ind] == 0 and defended[verify_ind] == 1:
                defended[verify_ind] = self.purify_and_predict(X_adv[verify_ind].unsqueeze(0), y[verify_ind].view([1]),
                                                               self.config['eot_defense_reps'], requires_grad=False)[0]
        return defended

    def eval_and_bpda_eot_grad(self, X_adv, y, defended, requires_grad=True):
        correct, attack_grad = self.purify_and_predict(X_adv, y, self.config['eot_attack_reps'], requires_grad)
        if self.config['eot_defense_reps'] > 0:
            defended = self.eot_defense_verification(X_adv, y, correct, defended)
        else:
            defended *= correct
        return defended, attack_grad

    def attack_batch(self, X, y):
        # get baseline accuracy for natural images
        defended = self.eval_and_bpda_eot_grad(X, y, torch.ones_like(y).bool(), False)[0]
        print('Baseline: {} of {}'.format(defended.sum(), len(defended)))

        class_batch = torch.zeros([self.config['adv_steps'] + 2, X.shape[0]]).bool()
        class_batch[0] = defended.cpu()
        ims_adv_batch = torch.zeros(X.shape)
        for ind in range(defended.nelement()):
            if defended[ind] == 0:
                ims_adv_batch[ind] = X[ind].cpu()

        X_adv = X.clone()

        # adversarial attacks on a single batch of images
        for step in range(self.config['adv_steps'] + 1):
            defended, attack_grad = self.eval_and_bpda_eot_grad(X_adv, y, defended)

            class_batch[step+1] = defended.cpu()
            for ind in range(defended.nelement()):
                if class_batch[step, ind] == 1 and defended[ind] == 0:
                    ims_adv_batch[ind] = X_adv[ind].cpu()

            # update adversarial images (except on final iteration so final adv images match final eval)
            if step < self.config['adv_steps']:
                X_adv = self.pgd_update(X_adv, attack_grad, X, self.config['adv_norm'], self.config['adv_eps'], self.config['adv_eta'])
                X_adv = X_adv.detach().clone()

            if step == 1 or step % self.config['log_freq'] == 0 or step == self.config['adv_steps']:
                print('Attack {} of {}   Batch defended: {} of {}'.
                      format(step, self.config['adv_steps'], int(torch.sum(defended).cpu().numpy()), X_adv.shape[0]))

            if int(torch.sum(defended).cpu().numpy()) == 0:
                print('Attack successfully to the batch!')
                break

        for ind in range(defended.nelement()):
            if defended[ind] == 1:
                ims_adv_batch[ind] = X_adv[ind].cpu()

        return class_batch, ims_adv_batch

    def attack_all(self, X, y, batch_size):
        class_path = torch.zeros([self.config['adv_steps'] + 2, 0]).bool()
        ims_adv = torch.zeros(0)

        n_batches = X.shape[0] // batch_size
        if n_batches == 0 and X.shape[0] > 0:
            n_batches = 1
        for counter in range(n_batches):
            X_batch = X[counter * batch_size:min((counter + 1) * batch_size, X.shape[0])].clone().to(X.device)
            y_batch = y[counter * batch_size:min((counter + 1) * batch_size, X.shape[0])].clone().to(X.device)

            class_batch, ims_adv_batch = self.attack_batch(X_batch.contiguous(), y_batch.contiguous())
            class_path = torch.cat((class_path, class_batch), dim=1)
            ims_adv = torch.cat((ims_adv, ims_adv_batch), dim=0)
            print(f'finished {counter}-th batch in attack_all')

            output_pil = tensor2image(ims_adv_batch[0])
            output_pil.save(f'purified_imgs/n000050.jpg')
        return class_path, ims_adv


def compute_n_params(model, return_str=True):
    tot = 0
    for p in model.parameters():
        w = 1
        for x in p.shape:
            w *= x
        tot += w
    if return_str:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


class Logger(object):
    """
    Redirect stderr to stdout, optionally print stdout to a file,
    and optionally force flushing on both stdout and the file.
    """

    def __init__(self, file_name: str = None, file_mode: str = "w", should_flush: bool = True):
        self.file = None

        if file_name is not None:
            self.file = open(file_name, file_mode)

        self.should_flush = should_flush
        self.stdout = sys.stdout
        self.stderr = sys.stderr

        sys.stdout = self
        sys.stderr = self

    def __enter__(self) -> "Logger":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()

    def write(self, text: str) -> None:
        """Write text to stdout (and a file) and optionally flush."""
        if len(text) == 0: # workaround for a bug in VSCode debugger: sys.stdout.write(''); sys.stdout.flush() => crash
            return

        if self.file is not None:
            self.file.write(text)

        self.stdout.write(text)

        if self.should_flush:
            self.flush()

    def flush(self) -> None:
        """Flush written text to both stdout and a file, if open."""
        if self.file is not None:
            self.file.flush()

        self.stdout.flush()

    def close(self) -> None:
        """Flush, close possible files, and remove stdout/stderr mirroring."""
        self.flush()

        # if using multiple loggers, prevent closing in wrong order
        if sys.stdout is self:
            sys.stdout = self.stdout
        if sys.stderr is self:
            sys.stderr = self.stderr

        if self.file is not None:
            self.file.close()


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def update_state_dict(state_dict, idx_start=9):

    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[idx_start:]  # remove 'module.0.' of dataparallel
        new_state_dict[name]=v

    return new_state_dict


def tensor2image(img_tensor):
    img_pil = transforms.ToPILImage()(img_tensor)
    return img_pil
# ------------------------------------------------------------------------
def get_overly_purified_img(model, x_orig, bs=1, device=torch.device('cuda:0')):

    n_batches = x_orig.shape[0] // bs
    for counter in range(n_batches):
        x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        input = x[0]
        input_pil = tensor2image(input)
        input_pil.save(f'original_imgs/{counter}.jpg')

        ### print purified image shape
        output = model(x)[0]
        print('purified image shape', output.shape)
        output = torch.clamp(output, 0, 1)
        output_pil = tensor2image(output)
        output_pil.save(f'purified_imgs/person.jpg')


def get_accuracy(model, x_orig, y_orig, bs=64, device=torch.device('cuda:0')):
    n_batches = x_orig.shape[0] // bs
    acc = 0.
    for counter in range(n_batches):
        x = x_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)
        y = y_orig[counter * bs:min((counter + 1) * bs, x_orig.shape[0])].clone().to(device)

        output = model(x)

        acc += (output.max(1)[1] == y).float().sum()

    return (acc / x_orig.shape[0]).item()


def get_image_classifier(classifier_name):
    class _Wrapper_ResNet(nn.Module):
        def __init__(self, resnet):
            super().__init__()
            self.resnet = resnet
            self.mu = torch.Tensor([0.485, 0.456, 0.406]).float().view(3, 1, 1)
            self.sigma = torch.Tensor([0.229, 0.224, 0.225]).float().view(3, 1, 1)

        def forward(self, x):
            x = (x - self.mu.to(x.device)) / self.sigma.to(x.device)
            return self.resnet(x)

    if 'imagenet' in classifier_name:
        if 'resnet18' in classifier_name:
            print('using imagenet resnet18...')
            model = models.resnet18(pretrained=True).eval()
        elif 'resnet50' in classifier_name:
            print('using imagenet resnet50...')
            model = models.resnet50(pretrained=True).eval()
        elif 'resnet101' in classifier_name:
            print('using imagenet resnet101...')
            model = models.resnet101(pretrained=True).eval()
        elif 'wideresnet-50-2' in classifier_name:
            print('using imagenet wideresnet-50-2...')
            model = models.wide_resnet50_2(pretrained=True).eval()
        elif 'deit-s' in classifier_name:
            print('using imagenet deit-s...')
            model = torch.hub.load('facebookresearch/deit:main', 'deit_small_patch16_224', pretrained=True).eval()
        else:
            raise NotImplementedError(f'unknown {classifier_name}')

        wrapper_resnet = _Wrapper_ResNet(model)

    elif 'cifar10' in classifier_name:
        if 'wideresnet-28-10' in classifier_name:
            print('using cifar10 wideresnet-28-10...')
            model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-28-10-at0' in classifier_name:
            print('using cifar10 wrn-28-10-at0...')
            model = load_model(model_name='Gowal2021Improving_28_10_ddpm_100m', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-28-10-at1' in classifier_name:
            print('using cifar10 wrn-28-10-at1...')
            model = load_model(model_name='Gowal2020Uncovering_28_10_extra', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-70-16-at0' in classifier_name:
            print('using cifar10 wrn-70-16-at0...')
            model = load_model(model_name='Gowal2021Improving_70_16_ddpm_100m', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-70-16-at1' in classifier_name:
            print('using cifar10 wrn-70-16-at1...')
            model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10',
                               threat_model='Linf')  # pixel in [0, 1]

        elif 'wrn-70-16-L2-at1' in classifier_name:
            print('using cifar10 wrn-70-16-L2-at1...')
            model = load_model(model_name='Rebuffi2021Fixing_70_16_cutmix_extra', dataset='cifar10',
                               threat_model='L2')  # pixel in [0, 1]

        elif 'wideresnet-70-16' in classifier_name:
            print('using cifar10 wideresnet-70-16 (dm_wrn-70-16)...')
            from robustbench.model_zoo.architectures.dm_wide_resnet import DMWideResNet, Swish
            model = DMWideResNet(num_classes=10, depth=70, width=16, activation_fn=Swish)  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/wresnet-76-10/weights-best.pt'
            print(f"=> loading wideresnet-70-16 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path)['model_state_dict']))
            model.eval()
            print(f"=> loaded wideresnet-70-16 checkpoint")

        elif 'resnet-50' in classifier_name:
            print('using cifar10 resnet-50...')
            from classifiers.cifar10_resnet import ResNet50
            model = ResNet50()  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/resnet-50/weights.pt'
            print(f"=> loading resnet-50 checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            print(f"=> loaded resnet-50 checkpoint")

        elif 'wrn-70-16-dropout' in classifier_name:
            print('using cifar10 wrn-70-16-dropout (standard wrn-70-16-dropout)...')
            from classifiers.cifar10_resnet import WideResNet_70_16_dropout
            model = WideResNet_70_16_dropout()  # pixel in [0, 1]

            model_path = 'pretrained/cifar10/wrn-70-16-dropout/weights.pt'
            print(f"=> loading wrn-70-16-dropout checkpoint '{model_path}'")
            model.load_state_dict(update_state_dict(torch.load(model_path), idx_start=7))
            model.eval()
            print(f"=> loaded wrn-70-16-dropout checkpoint")

        else:
            raise NotImplementedError(f'unknown {classifier_name}')

        wrapper_resnet = model

    elif 'celebahq' in classifier_name:
        attribute = classifier_name.split('__')[-1]  # `celebahq__Smiling`
        ckpt_path = f'diffpure_package/pretrained/celebahq/{attribute}/net_best.pth'
        from diffpure_package.classifiers.attribute_classifier import ClassifierWrapper
        model = ClassifierWrapper(attribute, ckpt_path=ckpt_path)
        wrapper_resnet = model
    else:
        raise NotImplementedError(f'unknown {classifier_name}')

    return wrapper_resnet


def load_data(args, adv_batch_size):
    if 'imagenet' in args.domain:
        val_dir = './dataset/imagenet_lmdb/val'  # using imagenet lmdb data
        val_transform = data.get_transform(args.domain, 'imval', base_size=224)
        val_data = data.imagenet_lmdb_dataset_sub(val_dir, transform=val_transform,
                                                  num_sub=args.num_sub, data_seed=args.data_seed)
        n_samples = len(val_data)
        val_loader = DataLoader(val_data, batch_size=n_samples, shuffle=False, pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(val_loader))
    elif 'cifar10' in args.domain:
        data_dir = './dataset'
        transform = transforms.Compose([transforms.ToTensor()])
        val_data = data.cifar10_dataset_sub(data_dir, transform=transform,
                                            num_sub=args.num_sub, data_seed=args.data_seed)
        n_samples = len(val_data)
        val_loader = DataLoader(val_data, batch_size=n_samples, shuffle=False, pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(val_loader))
    elif 'celebahq' in args.domain:
        data_dir = 'dataset/celeba-1024'
        attribute = args.classifier_name.split('__')[-1]  # `celebahq__Smiling`
        val_transform = data.get_transform('celebahq', 'imval')
        clean_dset = data.get_dataset('celebahq', 'val', attribute, root=data_dir, transform=val_transform,
                                      fraction=2, data_seed=args.data_seed)  # data_seed randomizes here
        loader = DataLoader(clean_dset, batch_size=adv_batch_size, shuffle=False,
                            pin_memory=True, num_workers=4)
        x_val, y_val = next(iter(loader))  # [0, 1], 256x256
    else:
        raise NotImplementedError(f'Unknown domain: {args.domain}!')

    print(f'x_val shape: {x_val.shape}')
    x_val, y_val = x_val.contiguous().requires_grad_(True), y_val.contiguous()
    print(f'x (min, max): ({x_val.min()}, {x_val.max()})')

    return x_val, y_val


class ResNet_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        # image classifier
        self.resnet = get_image_classifier(args.classifier_name).to(config.device)

    def purify(self, x):
        return x

    def forward(self, x, mode='purify_and_classify'):
        if mode == 'purify':
            out = self.purify(x)
        elif mode == 'classify':
            out = self.resnet(x)  # x in [0, 1]
        elif mode == 'purify_and_classify':
            x = self.purify(x)
            out = self.resnet(x)  # x in [0, 1]
        else:
            raise NotImplementedError(f'unknown mode: {mode}')
        return out


class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        
        # image classifier
        self.resnet = get_image_classifier(args.classifier_name).to(config.device)

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'sde':
            self.runner = RevGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'celebahq-ddpm':
            self.runner = Diffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    # use `counter` to record the the sampling time every 5 NFEs (note we hardcoded print freq to 5,
    # and you may want to change the freq)
    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def purify(self, x):
        counter = self.counter.item()
        if counter % 5 == 0:
            print(f'diffusion times: {counter}')

        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        start_time = time.time()
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        minutes, seconds = divmod(time.time() - start_time, 60)

        if 'imagenet' in self.args.domain:
            x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)

        if counter % 5 == 0:
            print(f'x shape (before diffusion models): {x.shape}')
            print(f'x shape (before resnet): {x_re.shape}')
            print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        self.counter += 1

        return (x_re + 1) * 0.5

    def forward(self, x, mode='purify'):
        if mode == 'purify':
            out = self.purify(x)
        elif mode == 'classify':
            out = self.resnet(x)  # x in [0, 1]
        elif mode == 'purify_and_classify':
            x = self.purify(x)
            out = self.resnet(x)  # x in [0, 1]
        else:
            raise NotImplementedError(f'unknown mode: {mode}')
        return out


def eval_bpda(config, model, x_val, adv_batch_size):
    ngpus = torch.cuda.device_count()
    model_ = model
    if ngpus > 1:
        model_ = model.module

    x_val = x_val.unsqueeze(0)
    x_val = x_val.to(config.device)

    # ------------------ apply the attack to sde_adv ------------------
    print(f'apply the bpda attack to sde_adv...')

    start_time = time.time()
    model_.reset_counter()
    model_.set_tag('no_adv')
    get_overly_purified_img(model, x_val, bs=adv_batch_size)

    adversary_sde = BPDA_EOT_Attack(model, adv_eps=args.adv_eps, eot_defense_reps=args.eot_defense_reps,
                                    eot_attack_reps=args.eot_attack_reps)

    start_time = time.time()
    model_.reset_counter()
    model_.set_tag()
    class_batch, ims_adv_batch = adversary_sde.attack_all(x_val, y_val, batch_size=adv_batch_size)
    # init_acc = float(class_batch[0, :].sum()) / class_batch.shape[1]
    # robust_acc = float(class_batch[-1, :].sum()) / class_batch.shape[1]
    #
    # print('init acc: {:.2%}, robust acc: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, robust_acc, time.time() - start_time))
    #
    # print(f'x_adv_sde shape: {ims_adv_batch.shape}')
    # torch.save([ims_adv_batch, y_val], f'{log_dir}/x_adv_sde_sd{args.seed}.pt')



PIL2tensor = transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor()
])

def run(args, config, img_path):
    middle_name = '_'.join([args.diffusion_type, 'bpda'])
    log_dir = os.path.join(args.image_folder, args.classifier_name, middle_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    # load model
    print('starting the model and loader...')
    model = SDE_Adv_Model(args, config)
    if ngpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.eval().to(config.device)

    # load data
    img_name = os.path.basename(img_path)[:-4] # peacock
    output_dir = os.path.dirname(img_path)

    # process on every img
    img_pil = Image.open(img_path)
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    image_tensor = PIL2tensor(img_pil).to(device)

    # eval classifier and sde_adv against bpda attack
    # eval_bpda(config, model, image_tensor, adv_batch_size)
    output = model(image_tensor.unsqueeze(0))
    output_pil = tensor2image(output[0])

    os.makedirs(f'{output_dir}_diffpure', exist_ok=True)
    output_pil.save(f'{output_dir}_diffpure/{img_name}_diffpure.png')

    logger.close()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument("--dataset_name", type=str, default='myfriends')
    parser.add_argument("--img_ids", type=str, nargs='+', default=["ziyi", 'qian', 'jiyan', 'weitsang'])
    parser.add_argument("--attacks", type=str, nargs='+', default=['metacloak', 'aspl', 'glaze', 'mist'])

    parser.add_argument('--config', type=str, default='celeba.yml', help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='celebahq-ddpm', help='[ddpm, sde, celebahq-ddpm]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')

    parser.add_argument('--eot_defense_reps', type=int, default=20)
    parser.add_argument('--eot_attack_reps', type=int, default=15)

    # adv
    parser.add_argument('--domain', type=str, default='celebahq', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--classifier_name', type=str, default='celebahq__Smiling', help='which classifier to use')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=1)

    parser.add_argument('--num_sub', type=int, default=1000, help='imagenet subset')
    parser.add_argument('--adv_eps', type=float, default=0.07)

    args = parser.parse_args()

    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config

def main(args, config, img_path):
    # if path is a folder containing images
    if os.path.isdir(img_path):
        img_paths = glob.glob(img_path + '/*')
        for p in img_paths:
            run(args, config, img_path=p)
    # if path is a single image
    else:
        run(args, config, img_path=img_path)


if __name__ == '__main__':
    args, config = parse_args_and_config()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
    for img_id in args.img_ids:
        for adv_algorithm in args.attacks:
            img_path = f'{args.dataset_name}/{img_id}_{adv_algorithm}'
            main(args, config, img_path)
