import collections
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from models.GAN import Generator
from PIL import Image

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class PerturbationTool():
    def __init__(self, args, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725, 
                 normalize_method='sample_wise_linf'):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        self.generator = Generator(args).cuda()
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.001)  
        np.random.seed(seed)
    

    def random_noise(self, noise_shape=[10, 3, 32, 32]):
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)
        return random_noise

    def stage1_min_min_attack(self, args, images, labels, copymodel, criterion):
        for _ in range(self.num_steps):
            noise = self.generator(images) 
            
            temp_class_noise = collections.defaultdict(list)
            currentbatch_class_noise = torch.zeros(*args.noise_shape)
            for i in range(len(noise)):
                temp_class_noise[labels[i].item()].append(noise[i])
            for key in temp_class_noise:
                currentbatch_class_noise[key] = torch.stack(temp_class_noise[key]).mean(dim=0)
            
            logits_ref = copymodel(images)

            batch_size = logits_ref.shape[0]
            mask = torch.zeros_like(logits_ref)
            mask[torch.arange(batch_size), labels] = float('-inf')
            masked_logits = logits_ref + mask
            _, labels_adv = masked_logits.max(dim=1)

            noise_relabel = torch.stack([currentbatch_class_noise[label_adv.item()] for label_adv in labels_adv]).cuda()
            noise_relabel = noise_relabel * args.epsilon
            perturb_img_relabel = torch.clamp(images + noise_relabel, 0, 1)

            logits_relabel = copymodel(perturb_img_relabel)
            loss_relabel = criterion(logits_relabel, labels_adv).requires_grad_()

            loss_G = loss_relabel
            
            self.optimizer_G.zero_grad()
            loss_G.backward(retain_graph=True)
            self.optimizer_G.step()

            noise_newly_optimized = self.generator(images)
            noise_newly_optimized = noise_newly_optimized * args.epsilon    

        return noise_newly_optimized.detach(), loss_relabel

    def stage2_min_min_attack(self, args, images, labels, copymodel, criterion):
            for _ in range(self.num_steps):
                noise = self.generator(images) 
                temp_class_noise = collections.defaultdict(list)
                currentbatch_class_noise = torch.zeros(*args.noise_shape)
                for i in range(len(noise)):
                    temp_class_noise[labels[i].item()].append(noise[i])
                for key in temp_class_noise:
                    currentbatch_class_noise[key] = torch.stack(temp_class_noise[key]).mean(dim=0)

                if 'CIFAR10' in args.train_data_type and 'CIFAR100' not in args.train_data_type:
                    num_classes = 10
                elif 'CIFAR100' in args.train_data_type:
                    num_classes = 100
                elif 'SVHN' in args.train_data_type:
                    num_classes = 10
                    
                labels_adv = torch.randint(0, num_classes, (labels.shape[0],), device=device, dtype=labels.dtype)
                same_indices = (labels_adv == labels)
                while same_indices.any():
                    labels_adv[same_indices] = torch.randint(0, num_classes, (same_indices.sum(),), device=device, dtype=labels.dtype)
                    same_indices = (labels_adv == labels)

                noise_relabel = torch.stack([currentbatch_class_noise[label_adv.item()] for label_adv in labels_adv]).cuda()
                noise_relabel = noise_relabel * args.epsilon
                perturb_img_relabel = torch.clamp(images + noise_relabel, 0, 1)

                logits_relabel = copymodel(perturb_img_relabel)
                loss_relabel = criterion(logits_relabel, labels_adv).requires_grad_()

                loss_G = loss_relabel
                
                self.optimizer_G.zero_grad()
                loss_G.backward(retain_graph=True)
                self.optimizer_G.step()

                noise_newly_optimized = self.generator(images)
                noise_newly_optimized = noise_newly_optimized * args.epsilon 

            return noise_newly_optimized.detach(), loss_relabel

    def stage3_min_min_attack(self, args, images, labels, copymodel, criterion):
            for _ in range(self.num_steps):
                noise = self.generator(images) 
                temp_class_noise = collections.defaultdict(list)
                currentbatch_class_noise = torch.zeros(*args.noise_shape)
                for i in range(len(noise)):
                    temp_class_noise[labels[i].item()].append(noise[i])
                for key in temp_class_noise:
                    currentbatch_class_noise[key] = torch.stack(temp_class_noise[key]).mean(dim=0)
                
                logits_ref = copymodel(images)

                _, labels_adv = logits_ref.min(dim=1)

                noise_relabel = torch.stack([currentbatch_class_noise[label_adv.item()] for label_adv in labels_adv]).cuda()

                noise_relabel = noise_relabel * args.epsilon
                perturb_img_relabel = torch.clamp(images + noise_relabel, 0, 1)

                logits_relabel = copymodel(perturb_img_relabel)
                loss_relabel = criterion(logits_relabel, labels_adv).requires_grad_()

                loss_G = loss_relabel
                
                self.optimizer_G.zero_grad()
                loss_G.backward(retain_graph=True)
                self.optimizer_G.step()

                noise_newly_optimized = self.generator(images)
                noise_newly_optimized = noise_newly_optimized * args.epsilon    

            return noise_newly_optimized.detach(), loss_relabel

    def _patch_noise_extend_to_img(self, noise, image_size=[3, 32, 32], patch_location='center'):
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

        if patch_location == 'center' or (h == w == x_len == y_len):
            x = h // 2
            y = w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, w - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
        else:
            raise('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, h)
        x2 = np.clip(x + x_len // 2, 0, h)
        y1 = np.clip(y - y_len // 2, 0, w)
        y2 = np.clip(y + y_len // 2, 0, w)
        if type(noise) is np.ndarray:
            pass
        else:
            mask[:, x1: x2, y1: y2] = noise.cpu().numpy()
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))
