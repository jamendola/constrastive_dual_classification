import torch
import torch.nn as nn
from torch.nn import functional as F
from datetime import datetime
import copy
from pytorch_lightning.strategies.ddp import DDPStrategy
from argparse import ArgumentParser
from thermaldata import Thermal, collate_fn
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from torchvision.transforms import Grayscale, Resize
from vgg_thermal_ae import VGGEncoder, VGGDecoder, EncoderLayer, EncoderBlock
from cond_conv import CondConv2d
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
import wandb
import itertools

class LEncoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super().__init__()

        if layers == 1:

            layer = LEncoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('0 EncoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = LEncoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = LEncoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = LEncoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d EncoderLayer' % i, layer)

        maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.add_module('%d MaxPooling' % layers, maxpool)

    def forward(self, x):

        for name, layer in self.named_children():
            x = layer(x)

        return x

class LEncoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super().__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(output_dim),
                nn.LeakyReLU(inplace=True),
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
                nn.LeakyReLU(inplace=True),
            )

    def forward(self, x):

        return self.layer(x)

class LVGGEncoder(nn.Module):

    def __init__(self, configs, enable_bn=False):
        super().__init__()

        if len(configs) != 5:
            raise ValueError("There should be 5 stage in VGG")

        self.conv1 = LEncoderBlock(input_dim=1, output_dim=64, hidden_dim=64, layers=configs[0], enable_bn=enable_bn)
        self.conv2 = LEncoderBlock(input_dim=64, output_dim=128, hidden_dim=128, layers=configs[1], enable_bn=enable_bn)
        self.conv3 = LEncoderBlock(input_dim=128, output_dim=256, hidden_dim=256, layers=configs[2], enable_bn=enable_bn)
        self.conv4 = LEncoderBlock(input_dim=256, output_dim=512, hidden_dim=512, layers=configs[3], enable_bn=enable_bn)
        self.conv5 = LEncoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=configs[4], enable_bn=enable_bn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

class LeakyVGGEncoder(nn.Module):

    def __init__(self, enable_bn=False, input_dimension=1):
        super().__init__()
        self.conv1 = LEncoderBlock(input_dim=input_dimension, output_dim=64, hidden_dim=64, layers=2, enable_bn=enable_bn)
        self.conv2 = LEncoderBlock(input_dim=64, output_dim=128, hidden_dim=128, layers=2, enable_bn=enable_bn)
        self.conv3 = LEncoderBlock(input_dim=128, output_dim=256, hidden_dim=256, layers=3, enable_bn=enable_bn)
        self.conv4 = LEncoderBlock(input_dim=256, output_dim=512, hidden_dim=512, layers=3, enable_bn=enable_bn)
        self.conv5 = LEncoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=3, enable_bn=enable_bn)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x

class LDecoderLayer(nn.Module):

    def __init__(self, input_dim, output_dim, enable_bn):
        super().__init__()

        if enable_bn:
            self.layer = nn.Sequential(
                nn.BatchNorm2d(input_dim),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )
        else:
            self.layer = nn.Sequential(
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(in_channels=input_dim, out_channels=output_dim, kernel_size=3, stride=1, padding=1),
            )

    def forward(self, x):
        #print('Debug layer', x.size())
        return self.layer(x)

class LDecoderBlock(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, layers, enable_bn=False):

        super().__init__()

        upsample = nn.ConvTranspose2d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=2, stride=2)

        self.add_module('0 UpSampling', upsample)

        if layers == 1:

            layer = LDecoderLayer(input_dim=input_dim, output_dim=output_dim, enable_bn=enable_bn)

            self.add_module('1 DecoderLayer', layer)

        else:

            for i in range(layers):

                if i == 0:
                    layer = LDecoderLayer(input_dim=input_dim, output_dim=hidden_dim, enable_bn=enable_bn)
                elif i == (layers - 1):
                    layer = LDecoderLayer(input_dim=hidden_dim, output_dim=output_dim, enable_bn=enable_bn)
                else:
                    layer = LDecoderLayer(input_dim=hidden_dim, output_dim=hidden_dim, enable_bn=enable_bn)

                self.add_module('%d DecoderLayer' % (i + 1), layer)

    def forward(self, x):

        for name, layer in self.named_children():
         #   print('Debug layers', x.size())
            x = layer(x)

        return x

class LeakyVGGDecoder(nn.Module):
    def __init__(self, output_dimension=1):
        super().__init__()
        self.conv1 = LDecoderBlock(input_dim=512, output_dim=512, hidden_dim=512, layers=3, enable_bn=True)
        self.conv2 = LDecoderBlock(input_dim=512, output_dim=256, hidden_dim=512, layers=3, enable_bn=True)
        self.conv3 = LDecoderBlock(input_dim=256, output_dim=128, hidden_dim=256, layers=3, enable_bn=True)
        self.conv4 = LDecoderBlock(input_dim=128, output_dim=64, hidden_dim=128, layers=2, enable_bn=True)
        self.conv5 = LDecoderBlock(input_dim=64, output_dim=output_dimension, hidden_dim=64, layers=2, enable_bn=True)
        # self.gate = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # x = self.gate(x)
        return x


class LightEncoder(nn.Module):
    def __init__(self, input_dimension=1):
        super().__init__()
        self.conv_a = nn.Sequential(
            nn.Conv2d(in_channels=input_dimension, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.conv_b = nn.Sequential(
            nn.Conv2d(in_channels=input_dimension, out_channels=64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.conv_c = nn.Sequential(
            nn.Conv2d(in_channels=input_dimension, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True))
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x_a = self.conv_a(x)
        x_b = self.conv_b(x)
        x_c = self.conv_c(x)
        x = torch.cat((x_a, x_b, x_c), 1)
        x = self.max_pool(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.conv3(x)
        return x

class CrossBatchNXENTLightEncoder(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        # self.gray_transformer = Grayscale(num_output_channels=1)
        self.lr = 0.005
        # VGG without Bn as AutoEncoder is hard to train
        self.thermal_encoder = LightEncoder(input_dimension=1)
        self.rgb_encoder = LightEncoder(input_dimension=3)
        self.contrastive_loss = losses.CrossBatchMemory(loss=losses.NTXentLoss(temperature=0.1), embedding_size=3136, memory_size=512)
        self.resize = Resize((224, 224))

    def forward(self, x, mode='thermal'):
        if mode == 'thermal':
            z = self.thermal_encoder(x)
        elif mode == 'rgb':
            z = self.rgb_encoder(x)
        return z

    def _run_step(self, x, mode):
        if mode == 'thermal':
            z = self.thermal_encoder(x)
        elif mode == 'rgb':
            z = self.rgb_encoder(x)
        return z

    def step(self, batch, batch_idx):
        x_color, x_thermal, bboxes, labels = batch
        x_color = self.resize(x_color)
        x_thermal = self.resize(x_thermal)

        z_rgb = self._run_step(x_color, mode='rgb')
        z_thermal = self._run_step(x_thermal, mode='thermal')

        # z_loss = F.l1_loss(z_gray, z_thermal)
        z_rgb_f = torch.flatten(z_rgb, start_dim=1)
        z_thermal_f = torch.flatten(z_thermal, start_dim=1)

        batch_size = z_rgb_f.size(0)

        #thermal-rgb pair pull force
        pair_labels = torch.arange(batch_size)
        embeddings = torch.cat([z_rgb_f, z_thermal_f], dim=0)
        pair_labels = torch.cat([pair_labels, pair_labels], dim=0)
        pair_loss = self.contrastive_loss(embeddings, pair_labels)

        #class grouping
        #TODO: expand bach for all detections, duplicanding embeddings when necessary
        # for label in labels:
        # org_labels = torch.cat(labels, dim=0)
        # repeat = torch.tensor(list(map(lambda x: x.size(0), labels))).to(org_labels)
        # # print('debug contrastive', org_labels, repeat)
        # rgb_embeddings = torch.repeat_interleave(z_rgb_f, repeat, dim=0)
        # class_pull_loss = self.contrastive_loss(rgb_embeddings, org_labels)

        loss = pair_loss
               # + class_pull_loss
        logs = {
            "pair_loss": pair_loss,
            # "class_pull_loss_rgb": class_pull_loss
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


class ContrastiveEncoder(LightningModule):

    def __init__(self, configs=[2, 2, 3, 3, 3]):
        super().__init__()
        self.save_hyperparameters()
        # self.gray_transformer = Grayscale(num_output_channels=1)
        self.lr = 0.005
        # VGG without Bn as AutoEncoder is hard to train
        self.thermal_encoder = LeakyVGGEncoder(input_dimension=1)
        self.rgb_encoder = LeakyVGGEncoder(input_dimension=3)
        self.contrastive_loss = losses.ContrastiveLoss()
        self.resize = Resize((224, 224))

    def forward(self, x, mode='thermal'):
        if mode == 'thermal':
            z = self.thermal_encoder(x)
        elif mode == 'rgb':
            z = self.rgb_encoder(x)
        return z

    def _run_step(self, x, mode):
        if mode == 'thermal':
            z = self.thermal_encoder(x)
        elif mode == 'rgb':
            z = self.rgb_encoder(x)
        return z

    def step(self, batch, batch_idx):
        x_color, x_thermal, bboxes, labels = batch
        x_color = self.resize(x_color)
        x_thermal = self.resize(x_thermal)

        z_rgb = self._run_step(x_color, mode='rgb')
        z_thermal = self._run_step(x_thermal, mode='thermal')

        # z_loss = F.l1_loss(z_gray, z_thermal)
        z_rgb_f = torch.flatten(z_rgb, start_dim=1)
        z_thermal_f = torch.flatten(z_thermal, start_dim=1)

        batch_size = z_rgb_f.size(0)

        #thermal-rgb pair pull force
        pair_labels = torch.arange(batch_size)
        embeddings = torch.cat([z_rgb_f, z_thermal_f], dim=0)
        pair_labels = torch.cat([pair_labels, pair_labels], dim=0)
        pair_loss = self.contrastive_loss(embeddings, pair_labels)

        #class grouping
        #TODO: expand bach for all detections, duplicanding embeddings when necessary
        # for label in labels:
        # org_labels = torch.cat(labels, dim=0)
        # repeat = torch.tensor(list(map(lambda x: x.size(0), labels))).to(org_labels)
        # # print('debug contrastive', org_labels, repeat)
        # rgb_embeddings = torch.repeat_interleave(z_rgb_f, repeat, dim=0)
        # class_pull_loss = self.contrastive_loss(rgb_embeddings, org_labels)

        loss = pair_loss
               # + class_pull_loss
        logs = {
            "pair_loss": pair_loss,
            # "class_pull_loss_rgb": class_pull_loss
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)

class ClassContrastiveEncoder(ContrastiveEncoder):
    def __init__(self):
        super().__init__()

    def step(self, batch, batch_idx):
        x_color, x_thermal, bboxes, labels = batch
        x_color = self.resize(x_color)
        x_thermal = self.resize(x_thermal)

        z_rgb = self._run_step(x_color, mode='rgb')
        z_thermal = self._run_step(x_thermal, mode='thermal')

        # z_loss = F.l1_loss(z_gray, z_thermal)
        z_rgb_f = torch.flatten(z_rgb, start_dim=1)
        z_thermal_f = torch.flatten(z_thermal, start_dim=1)

        batch_size = z_rgb_f.size(0)

        #thermal-rgb pair pull force
        pair_labels = torch.arange(batch_size)
        embeddings = torch.cat([z_rgb_f, z_thermal_f], dim=0)
        pair_labels = torch.cat([pair_labels, pair_labels], dim=0)
        pair_loss = self.contrastive_loss(embeddings, pair_labels)

        #class grouping
        #TODO: expand bach for all detections, duplicanding embeddings when necessary
        # for label in labels:
        org_labels = torch.cat(labels, dim=0)
        # print("DEBUG org_labels", org_labels)
        # repeat = torch.tensor(list(map(lambda x: x.size(0), labels))).to(org_labels)
        # print('debug contrastive', org_labels, repeat)
        # rgb_embeddings = torch.repeat_interleave(z_rgb_f, repeat, dim=0)
        rgb_class_pull_loss = self.contrastive_loss(z_rgb_f, org_labels)

        # org_labels = torch.cat(labels, dim=0)
        # repeat = torch.tensor(list(map(lambda x: x.size(0), labels))).to(org_labels)
        # print('debug contrastive', org_labels, repeat)
        # rgb_embeddings = torch.repeat_interleave(z_rgb_f, repeat, dim=0)
        thermal_class_pull_loss = self.contrastive_loss(z_thermal_f, org_labels)

        loss = 0.05*pair_loss + rgb_class_pull_loss + thermal_class_pull_loss
        logs = {
            "pair_loss": pair_loss,
            "rgb_class_pull_loss_rgb": rgb_class_pull_loss,
            "thermal_class_pull_loss": thermal_class_pull_loss
        }
        return loss, logs

class ClassContrastiveEncoderB(CrossBatchNXENTLightEncoder):
    def __init__(self):
        super().__init__()

    def step(self, batch, batch_idx):
        x_color, x_thermal, bboxes, labels = batch
        x_color = self.resize(x_color)
        x_thermal = self.resize(x_thermal)

        z_rgb = self._run_step(x_color, mode='rgb')
        z_thermal = self._run_step(x_thermal, mode='thermal')

        # z_loss = F.l1_loss(z_gray, z_thermal)
        z_rgb_f = torch.flatten(z_rgb, start_dim=1)
        z_thermal_f = torch.flatten(z_thermal, start_dim=1)

        batch_size = z_rgb_f.size(0)

        #thermal-rgb pair pull force
        pair_labels = torch.arange(batch_size)
        embeddings = torch.cat([z_rgb_f, z_thermal_f], dim=0)
        pair_labels = torch.cat([pair_labels, pair_labels], dim=0)
        pair_loss = self.contrastive_loss(embeddings, pair_labels)

        #class grouping
        #TODO: expand bach for all detections, duplicanding embeddings when necessary
        # for label in labels:
        org_labels = torch.cat(labels, dim=0)
        # print("DEBUG org_labels", org_labels)
        # repeat = torch.tensor(list(map(lambda x: x.size(0), labels))).to(org_labels)
        # print('debug contrastive', org_labels, repeat)
        # rgb_embeddings = torch.repeat_interleave(z_rgb_f, repeat, dim=0)
        rgb_class_pull_loss = self.contrastive_loss(z_rgb_f, org_labels)

        # org_labels = torch.cat(labels, dim=0)
        # repeat = torch.tensor(list(map(lambda x: x.size(0), labels))).to(org_labels)
        # print('debug contrastive', org_labels, repeat)
        # rgb_embeddings = torch.repeat_interleave(z_rgb_f, repeat, dim=0)
        thermal_class_pull_loss = self.contrastive_loss(z_thermal_f, org_labels)

        loss = 0.05*pair_loss + rgb_class_pull_loss + thermal_class_pull_loss
        logs = {
            "pair_loss": pair_loss,
            "rgb_class_pull_loss_rgb": rgb_class_pull_loss,
            "thermal_class_pull_loss": thermal_class_pull_loss
        }
        return loss, logs

class SingleContrastiveEncoder(LightningModule):

    def __init__(self, configs=[2, 2, 3, 3, 3]):
        super().__init__()
        self.save_hyperparameters()
        # self.gray_transformer = Grayscale(num_output_channels=1)
        self.lr = 0.005
        # VGG without Bn as AutoEncoder is hard to train
        self.encoder = LeakyVGGEncoder(input_dimension=1)
        self.contrastive_loss = losses.NPairsLoss()
        self.resize = Resize((224, 224))
        self.gray_transformer = Grayscale(num_output_channels=1)


    def forward(self, x):
        z = self.encoder(x)
        return z

    def _run_step(self, x):
        z = self.encoder(x)
        return z

    def step(self, batch, batch_idx):
        x_color, x_thermal, bboxes, labels = batch
        x_color = self.resize(x_color)
        x_color = self.gray_transformer(x_color)
        x_thermal = self.resize(x_thermal)

        z_rgb = self._run_step(x_color)
        z_thermal = self._run_step(x_thermal)

        # z_loss = F.l1_loss(z_gray, z_thermal)
        z_rgb_f = torch.flatten(z_rgb, start_dim=1)
        z_thermal_f = torch.flatten(z_thermal, start_dim=1)

        batch_size = z_rgb_f.size(0)

        #thermal-rgb pair pull force
        pair_labels = torch.arange(batch_size)
        embeddings = torch.cat([z_rgb_f, z_thermal_f], dim=0)
        pair_labels = torch.cat([pair_labels, pair_labels], dim=0)
        pair_loss = self.contrastive_loss(embeddings, pair_labels)

        #class grouping
        #TODO: expand bach for all detections, duplicanding embeddings when necessary
        # for label in labels:
        # org_labels = torch.cat(labels, dim=0)
        # repeat = torch.tensor(list(map(lambda x: x.size(0), labels))).to(org_labels)
        # # print('debug contrastive', org_labels, repeat)
        # rgb_embeddings = torch.repeat_interleave(z_rgb_f, repeat, dim=0)
        # class_pull_loss = self.contrastive_loss(rgb_embeddings, org_labels)

        loss = pair_loss
               # + class_pull_loss
        logs = {
            "pair_loss": pair_loss,
            # "class_pull_loss_rgb": class_pull_loss
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)


class ContrastiveAutoEncoder(LightningModule):

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        # self.gray_transformer = Grayscale(num_output_channels=1)
        self.lr = 0.005
        # VGG without Bn as AutoEncoder is hard to train
        self.thermal_encoder = LeakyVGGEncoder(input_dimension=1)
        self.rgb_encoder = LeakyVGGEncoder(input_dimension=3)
        self.thermal_decoder = LeakyVGGDecoder(output_dimension=1)
        self.rgb_decoder = LeakyVGGDecoder(output_dimension=3)
        self.contrastive_loss = losses.NTXentLoss()
        self.resize = Resize((224, 224))

    def forward(self, x, mode='thermal'):
        if mode == 'thermal':
            z = self.thermal_encoder(x)
            x_hat = self.thermal_decoder(z)
        elif mode == 'rgb':
            z = self.rgb_encoder(x)
            x_hat = self.rgb_decoder
        return z, x_hat

    def _run_step(self, x, mode):
        if mode == 'thermal':
            z = self.thermal_encoder(x)
            x_hat = self.thermal_decoder(z)
        elif mode == 'rgb':
            z = self.rgb_encoder(x)
            x_hat = self.rgb_decoder(z)
        return z, x_hat

    def step(self, batch, batch_idx):
        x_color, x_thermal, bboxes, labels = batch
        x_color = self.resize(x_color)
        x_thermal = self.resize(x_thermal)

        z_rgb, x_hat_rgb = self._run_step(x_color, mode='rgb')
        z_thermal, x_hat_thermal = self._run_step(x_thermal, mode='thermal')

        rec_loss = F.l1_loss(x_hat_thermal, x_thermal) + F.l1_loss(x_hat_rgb, x_color)


        # z_loss = F.l1_loss(z_gray, z_thermal)
        z_rgb_f = torch.flatten(z_rgb, start_dim=1)
        z_thermal_f = torch.flatten(z_thermal, start_dim=1)

        batch_size = z_rgb_f.size(0)

        #thermal-rgb pair pull force
        pair_labels = torch.arange(batch_size)
        embeddings = torch.cat([z_rgb_f, z_thermal_f], dim=0)
        pair_labels = torch.cat([pair_labels, pair_labels], dim=0)
        pair_loss = self.contrastive_loss(embeddings, pair_labels)

        #class grouping
        #TODO: expand bach for all detections, duplicanding embeddings when necessary
        # for label in labels:
        # org_labels = torch.cat(labels, dim=0)
        # repeat = torch.tensor(list(map(lambda x: x.size(0), labels))).to(org_labels)
        # # print('debug contrastive', org_labels, repeat)
        # rgb_embeddings = torch.repeat_interleave(z_rgb_f, repeat, dim=0)
        # class_pull_loss = self.contrastive_loss(rgb_embeddings, org_labels)

        loss = pair_loss + rec_loss
               # + class_pull_loss
        logs = {
            "pair_loss": pair_loss,
            "rec_loss": rec_loss,
            "total_loss": loss
            # "class_pull_loss_rgb": class_pull_loss
        }
        return loss, logs

    def training_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in logs.items()}, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logs = self.step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in logs.items()}, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.lr)