import argparse
import collections
import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import shutil
import time
import dataset
import mlconfig
import toolbox
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import util
import madrys
import numpy as np
from evaluator import Evaluator
from tqdm import tqdm
from trainer import Trainer
import higher

mlconfig.register(madrys.MadrysLoss)


# General Options
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--version', type=str, default="resnet18")
parser.add_argument('--pretrained_version', type=str, default="resnet18", help='Pre-trained surrogate model architecture')
parser.add_argument('--exp_name', type=str, default="experiment/")
parser.add_argument('--config_path', type=str, default='configs/cifar10')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--epoch', default=60, type=int)
# Datasets Options
parser.add_argument('--train_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--eval_batch_size', default=512, type=int, help='perturb step size')
parser.add_argument('--num_of_workers', default=6, type=int, help='workers for loader')
parser.add_argument('--train_data_type', type=str, default='CIFAR10')
parser.add_argument('--train_data_path', type=str, default='datasets')
parser.add_argument('--test_data_type', type=str, default='CIFAR10')
parser.add_argument('--test_data_path', type=str, default='datasets')
# Perturbation Options
parser.add_argument('--universal_train_portion', default=0.5, type=float, help='only valid when args.use_subset is True')
parser.add_argument('--universal_stop_error', default=0.1, type=float)
parser.add_argument('--train_step', default=1, type=int)
parser.add_argument('--use_subset', action='store_true', default=False)
parser.add_argument('--attack_type', default='min-min', type=str, help='Attack type')
parser.add_argument('--perturb_type', default='classwise', type=str, help='Perturb type')
parser.add_argument('--patch_location', default='center', type=str, choices=['center', 'random'], help='Location of the noise')
parser.add_argument('--noise_shape', default=[10, 3, 32, 32], nargs='+', type=int, help='noise shape')
parser.add_argument('--epsilon', default=8, type=float, help='perturbation')
parser.add_argument('--num_steps', default=1, type=int, help='perturb number of steps')
parser.add_argument('--step_size', default=0.8, type=float, help='perturb step size')
parser.add_argument('--random_start', action='store_true', default=False)
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for surrogate model')
parser.add_argument('--use_pretrained_surrogate', action='store_true', default=False) 
parser.add_argument('--pretrained_type', type=str, default='') 
parser.add_argument('--meta_warmup_step', default=1, type=int, help='warm up steps for surrogate model')


args = parser.parse_args()

if args.use_subset:
    args.universal_train_target = 'train_subset'
else:
    args.universal_train_target = 'train_dataset'

# Convert Eps
args.epsilon = args.epsilon / 255      # noise bound
args.step_size = args.step_size / 255  

# Set up Experiments - Fix datetime format to be file-system friendly
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:25] 
args.exp_name = args.exp_name + args.perturb_type + '/noise_generation_' + current_time

exp_path = args.exp_name
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
generator_path_file = os.path.join(args.exp_name, 'generator')
util.build_dirs(exp_path)

logger = util.setup_logger(name=args.version, log_file=log_file_path + ".log")

# CUDA Options
logger.info("PyTorch Version: %s" % (torch.__version__))
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda')
    device_list = [torch.cuda.get_device_name(i) for i in range(0, torch.cuda.device_count())]
    logger.info("GPU List: %s" % (device_list))
else:
    device = torch.device('cpu')

# Load Exp Configs
config_file = os.path.join(args.config_path, args.version)+'.yaml'
config = mlconfig.load(config_file)
config['optimizer']['lr'] = args.lr
for key in config:
    logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version  +'.yaml'))

def universal_perturbation(args, noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, class_noise_each_epoch, ENV, data_loader):
    data_iter = iter(data_loader[args.universal_train_target])

    logger.info('=' * 20 + 'Searching Universal Perturbation' + '=' * 20)
    if hasattr(model, 'classify'):
        model.classify = True    

    epoch_count = 0
    logger.info("---- STAGE 1 Relabel to Hard Negative Classes----")
    for i in range(args.epoch):
        if epoch_count == 30:
            logger.info("---- STAGE 2 Relabel to Random Classes----")
        elif epoch_count == 60:
            logger.info("---- STAGE 3 Relabel to Most Dissimilar Classes----")
        try:
            (images, labels) = next(data_iter)
        except:
            data_iter = iter(data_loader[args.universal_train_target])
            (images, labels) = next(data_iter)
        images, labels = images.to(device), labels.to(device)
        # Train Batch for min-min noise
        with higher.innerloop_ctx(model, optimizer, copy_initial_weights=True) as (copymodel, diff_optimizer):   
            for j in range(0, args.train_step):
                # Add Class-wise Noise to each sample
                train_copy_surrogate_imgs = []
                for i, (image, label) in enumerate(zip(images, labels)):
                    noise = class_noise_each_epoch[label.item()]  
                    mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=image.shape, patch_location=args.patch_location)
                    train_copy_surrogate_imgs.append(image + class_noise)  
                # Train
                copymodel.train()
                for param in model.parameters():
                    param.requires_grad = True
                log_payload = trainer.train_batch(torch.stack(train_copy_surrogate_imgs).to(device), labels, copymodel, diff_optimizer, meta_train=True)  
                loss_classifier_copy_surrogate = log_payload['loss']

            classwise_noise_all = []  
            for i, (real_images, real_labels) in tqdm(enumerate(data_loader[args.universal_train_target]), total=len(data_loader[args.universal_train_target])):
                real_images, real_labels = real_images.to(device), real_labels.to(device)

                # Add Class-wise Noise to each sample
                batch_noise, mask_cord_list = [], []

                for i, (real_image, label_adv) in enumerate(zip(real_images, real_labels)):
                    noise = class_noise_each_epoch[label_adv.item()]
                    mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=real_image.shape, patch_location=args.patch_location)
                    batch_noise.append(class_noise)
                    mask_cord_list.append(mask_cord)

                batch_noise = torch.stack(batch_noise).to(device)  
                if args.attack_type == 'min-min':    
                    if epoch_count < 30:
                        noise_outer, loss_relabel = noise_generator.stage1_min_min_attack(args, real_images, real_labels, copymodel, criterion)
                    elif epoch_count < 60:
                        noise_outer, loss_relabel = noise_generator.stage2_min_min_attack(args, real_images, real_labels, copymodel, criterion)
                    else:
                        noise_outer, loss_relabel = noise_generator.stage3_min_min_attack(args, real_images, real_labels, copymodel, criterion)
                else:
                    raise('Invalid attack')

                class_noise_eta_outer = collections.defaultdict(list)
                for i in range(len(noise_outer)):
                    x1, x2, y1, y2 = mask_cord_list[i]
                    delta_outer = noise_outer[i][:, x1: x2, y1: y2]
                    class_noise_eta_outer[real_labels[i].item()].append(delta_outer.detach().cpu())

                for key in class_noise_eta_outer:
                    delta_outer = torch.stack(class_noise_eta_outer[key]).mean(dim=0) - class_noise_each_epoch[key]
                    class_noise = class_noise_each_epoch[key]
                    class_noise += delta_outer
                    class_noise_each_epoch[key] = torch.clamp(class_noise, -args.epsilon, args.epsilon)
            torch.cuda.empty_cache()  

        for k in range(0, args.meta_warmup_step):
            train_surrogate_imgs = []
            for i, (image, label) in enumerate(zip(images, labels)):
                with torch.no_grad():
                    noise = class_noise_each_epoch[label.item()]  
                    mask_cord, class_noise = noise_generator._patch_noise_extend_to_img(noise, image_size=image.shape, patch_location=args.patch_location)
                    train_surrogate_imgs.append(image + class_noise) 
            # Train
            model.train()
            log_payload = trainer.train_batch(torch.stack(train_surrogate_imgs).to(device), labels, model, optimizer, meta_train=False)
            loss_classifier_surrogate = log_payload['loss']

        logger.info('loss_classifier_copy_surrogate: {:.4f} loss_classifier_surrogate: {:.4f} loss_relabel: {:.4f}'.format(loss_classifier_copy_surrogate.item(), loss_classifier_surrogate.item(), loss_relabel.item()))
        class_noise_each_epoch = class_noise_each_epoch.detach()
        ENV['random_noise'] = class_noise_each_epoch

        torch.cuda.empty_cache()  
        epoch_count += 1

    surrogate_filename = os.path.join(checkpoint_path_file, args.version + '.pth')
    os.makedirs(os.path.dirname(surrogate_filename), exist_ok=True)
    torch.save(model.state_dict(), surrogate_filename)
    logger.info('Model Saved at %s', surrogate_filename)

    return class_noise_each_epoch

def main():
    # Setup ENV
    datasets_generator = dataset.DatasetGenerator(args=args, train_batch_size=args.train_batch_size,
                                                  eval_batch_size=args.eval_batch_size,
                                                  train_data_type=args.train_data_type,
                                                  train_data_path=args.train_data_path,
                                                  test_data_type=args.test_data_type,
                                                  test_data_path=args.test_data_path,
                                                  num_of_workers=args.num_of_workers,
                                                  seed=args.seed)
    if not args.use_pretrained_surrogate:
        logger.info('==> Loading Training-from-scratch Surrogate..')
        model = config.model().to(device)
        for param in model.parameters():
            param.requires_grad = True
    else:
        # Load the pretrained ResNet18 model
        if args.pretrained_type == 'imagenet':
            logger.info('==> Loading Imagenet Pre-trained Surrogate..')
            if args.pretrained_version == 'resnet18':
                model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).to(device)

    # Determine num_classes based on the dataset
    if 'CIFAR10' in args.train_data_type and 'CIFAR100' not in args.train_data_type:
        num_classes = 10
    elif 'CIFAR100' in args.train_data_type:
        num_classes = 100
    elif 'SVHN' in args.train_data_type:
        num_classes = 10
    else:
        raise ValueError("Unsupported dataset type: {}".format(args.train_data_type))

    if args.pretrained_type == 'imagenet':
        if 'CIFAR10' in args.train_data_type or 'CIFAR100' in args.train_data_type or 'SVHN' in args.train_data_type:
            logger.info('==> Modifying the last layer to have %d outputs..' % num_classes)
            if hasattr(model, 'fc'):  # ResNet
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, num_classes)
            model = model.to(device)

            for param in model.parameters():
                param.requires_grad = True
        else:
            logger.info("Unsupported dataset type for pre-trained model: {}".format(args.train_data_type))
            exit()

    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    criterion = config.criterion()
    if args.perturb_type == 'classwise':
        if args.use_subset:  
            data_loader = datasets_generator._split_validation_set(train_portion=args.universal_train_portion,
                                                                   train_shuffle=True, train_drop_last=True)
            train_target = 'train_subset'
        else:
            data_loader = datasets_generator.getDataLoader(train_shuffle=True, train_drop_last=True)
            train_target = 'train_dataset'

    trainer = Trainer(criterion, data_loader, logger, config, target=train_target)
    evaluator = Evaluator(data_loader, logger, config)
    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'current_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': []}

    if args.data_parallel:  
        model = torch.nn.DataParallel(model)

    if args.load_model:  
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=scheduler)
        ENV = checkpoint['ENV']
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file))

    noise_generator = toolbox.PerturbationTool(args,
                                               epsilon=args.epsilon,
                                               num_steps=args.num_steps,
                                               step_size=args.step_size)

    if args.attack_type == 'random':
        noise = noise_generator.random_noise(noise_shape=args.noise_shape)
        torch.save(noise, os.path.join(args.exp_name, 'perturbation.pt'))
        logger.info(noise)
        logger.info(noise.shape)
        logger.info('Noise saved at %s' % (os.path.join(args.exp_name, 'perturbation.pt')))
    elif args.attack_type == 'min-min' or args.attack_type == 'min-max':
        if args.random_start:  
            random_noise = noise_generator.random_noise(noise_shape=args.noise_shape)
        else:
            random_noise = torch.zeros(*args.noise_shape)   
        if args.perturb_type == 'classwise':
            noise = universal_perturbation(args, noise_generator, trainer, evaluator, model, criterion, optimizer, scheduler, random_noise, ENV, data_loader)
        torch.save(noise, os.path.join(args.exp_name, 'perturbation.pt'))
        logger.info(noise)
        logger.info(noise.shape)
        logger.info('Noise saved at %s' % (os.path.join(args.exp_name, 'perturbation.pt')))

        # Save Model
        net_G = noise_generator.generator 
        filename = generator_path_file + '.pth'
        torch.save(net_G.state_dict(), filename)
        logger.info('Generator Saved at %s', filename)
    else:
        raise('Not implemented yet')
    return


if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)
