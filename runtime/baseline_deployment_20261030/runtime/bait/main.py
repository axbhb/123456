import argparse
import datetime
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import shutil
import time
import numpy as np
import dataset
import mlconfig
import torch
import util
import madrys
import models
from evaluator import Evaluator
from trainer import Trainer
import torchvision
import torch.nn as nn
mlconfig.register(madrys.MadrysLoss)

def str2bool(v):
    return v.lower() in ('true', '1')

# General Options
parser = argparse.ArgumentParser(description='ClasswiseNoise')
parser.add_argument('--seed', type=int, default=0, help='seed')
parser.add_argument('--version', type=str, default="resnet18")
parser.add_argument('--pretrained_version', type=str, default="resnet18")
parser.add_argument('--exp_name', type=str, default="experiment/")
parser.add_argument('--config_path', type=str, default='configs/cifar10')
parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--train', default=True)
parser.add_argument('--save_model', action='store_true', default=False) 
# Datasets Options
parser.add_argument('--train_face', action='store_true', default=False)
parser.add_argument('--train_portion', default=1.0, type=float)
parser.add_argument('--train_batch_size', default=128, type=int, help='perturb step size')
parser.add_argument('--eval_batch_size', default=256, type=int, help='perturb step size')
parser.add_argument('--num_of_workers', default=8, type=int, help='workers for loader')
parser.add_argument('--train_data_type', type=str, default='PoisonCIFAR10')
parser.add_argument('--test_data_type', type=str, default='CIFAR10')
parser.add_argument('--train_data_path', type=str, default='datasets')
parser.add_argument('--test_data_path', type=str, default='datasets')
parser.add_argument('--perturb_type', default='classwise', type=str, choices=['classwise', 'samplewise'], help='Perturb type')
parser.add_argument('--patch_location', default='center', type=str, choices=['center', 'random'], help='Location of the noise')
parser.add_argument('--poison_rate', default=1.0, type=float)
parser.add_argument('--epsilon', default=8, type=float, help='perturbation')
parser.add_argument('--perturb_tensor_filepath', default='', type=str)
parser.add_argument('--generator_filepath', default='', type=str)
parser.add_argument('--use_generator', action='store_true', default=False)   
parser.add_argument('--lr', default=0.001, type=float, help='learning rate for finetuning')
parser.add_argument('--use_pretrained_model', action='store_true', default=False) 
parser.add_argument('--pretrained_type', type=str, default='imagenet') 
args = parser.parse_args()

# Set up Experiments - Fix datetime format to be file-system friendly
current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:25] 
args.exp_name = args.exp_name + args.perturb_type + '/model_training_' + current_time

args.epsilon = args.epsilon / 255      # noise bound
exp_path = args.exp_name
log_file_path = os.path.join(exp_path, args.version)
checkpoint_path = os.path.join(exp_path, 'checkpoints')
checkpoint_path_file = os.path.join(checkpoint_path, args.version)
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
config.set_immutable()
for key in config:
    logger.info("%s: %s" % (key, config[key]))
shutil.copyfile(config_file, os.path.join(exp_path, args.version+'.yaml'))


def train(args, starting_epoch, model, atk, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader):
    for epoch in range(starting_epoch, config.epochs):
        logger.info("")
        logger.info("="*20 + "Training Epoch %d" % (epoch) + "="*20)

        # Train
        ENV['global_step'] = trainer.train(args, epoch, model, atk, criterion, optimizer)
        ENV['train_history'].append(trainer.acc_meters.avg*100)
        scheduler.step()

        # Eval
        logger.info("="*20 + "Eval Epoch %d" % (epoch) + "="*20)
        is_best = False
        if not args.train_face:
            evaluator.eval(epoch, model)
            payload = ('Eval Loss:%.4f\tEval acc: %.2f' % (evaluator.loss_meters.avg, evaluator.acc_meters.avg*100))
            logger.info(payload)
            ENV['eval_history'].append(evaluator.acc_meters.avg*100)
            ENV['curren_acc'] = evaluator.acc_meters.avg*100
            ENV['cm_history'].append(evaluator.confusion_matrix.cpu().numpy().tolist())
            # Reset Stats
            trainer._reset_stats()
            evaluator._reset_stats()
        else:
            pass

    torch.save(model.state_dict(), os.path.join(exp_path, 'resnet18.pth'))
    logger.info('ResNet Model Saved at %s', os.path.join(exp_path, 'resnet18.pth'))

    return


def main():
    if not args.use_pretrained_model:
        logger.info('==> Loading Training-from-scratch Model..')
        model = config.model().to(device)
    else:
        # Load the pretrained ResNet18 model
        if args.pretrained_type == 'imagenet':
            logger.info('==> Loading Imagenet Pre-trained ResNet18 Model..')
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

        # Modify the last layer to have the correct number of outputs
        if 'CIFAR10' in args.train_data_type or 'CIFAR100' in args.train_data_type or 'SVHN' in args.train_data_type:
            if args.pretrained_type == 'imagenet':
                logger.info('==> Modifying the last layer to have %d outputs..' % num_classes)
                if hasattr(model, 'fc'):  # ResNet
                    num_ftrs = model.fc.in_features
                    model.fc = nn.Linear(num_ftrs, num_classes)
                model = model.to(device)

        logger.info('Training the entire model end-to-end')
        for param in model.parameters():
            param.requires_grad = True

    datasets_generator = config.dataset(args=args,
                                        train_data_type=args.train_data_type,
                                        train_data_path=args.train_data_path,
                                        test_data_type=args.test_data_type,
                                        test_data_path=args.test_data_path,
                                        train_batch_size=args.train_batch_size,
                                        eval_batch_size=args.eval_batch_size,
                                        num_of_workers=args.num_of_workers,
                                        poison_rate=args.poison_rate,
                                        perturb_type=args.perturb_type,
                                        patch_location=args.patch_location,
                                        perturb_tensor_filepath=args.perturb_tensor_filepath,
                                        seed=args.seed,
                                        use_generator=args.use_generator)
    logger.info('Training Dataset: %s' % str(datasets_generator.datasets['train_dataset']))
    logger.info('Test Dataset: %s' % str(datasets_generator.datasets['test_dataset']))

    if args.train_portion == 1.0:
        data_loader = datasets_generator.getDataLoader()
        train_target = 'train_dataset'
    else:
        train_target = 'train_subset'
        data_loader = datasets_generator._split_validation_set(args.train_portion,
                                                               train_shuffle=True,
                                                               train_drop_last=True)

    logger.info("param size = %fMB", util.count_parameters_in_MB(model))
    optimizer = config.optimizer(model.parameters())
    scheduler = config.scheduler(optimizer)
    criterion = config.criterion()
    trainer = Trainer(criterion, data_loader, logger, config, target=train_target)
    evaluator = Evaluator(data_loader, logger, config)

    starting_epoch = 0
    ENV = {'global_step': 0,
           'best_acc': 0.0,
           'current_acc': 0.0,
           'best_pgd_acc': 0.0,
           'train_history': [],
           'eval_history': [],
           'pgd_eval_history': [],
           'genotype_list': [],
           'cm_history': []}

    if args.load_model:
        checkpoint = util.load_model(filename=checkpoint_path_file,
                                     model=model,
                                     optimizer=optimizer,
                                     alpha_optimizer=None,
                                     scheduler=scheduler)
        starting_epoch = checkpoint['epoch']
        ENV = checkpoint['ENV']
        trainer.global_step = ENV['global_step']
        logger.info("File %s loaded!" % (checkpoint_path_file))

    if args.data_parallel:
        model = torch.nn.DataParallel(model)

    if args.train:
        train(args, starting_epoch, model, None, optimizer, scheduler, criterion, trainer, evaluator, ENV, data_loader)

        # Save ResNet model
        if args.save_model:
            torch.save(model.state_dict(), os.path.join(exp_path, 'resnet18.pth'))
            logger.info('ResNet Model Saved at %s', os.path.join(exp_path, 'resnet18.pth'))


if __name__ == '__main__':
    for arg in vars(args):
        logger.info("%s: %s" % (arg, getattr(args, arg)))
    start = time.time()
    main()
    end = time.time()
    cost = (end - start) / 86400
    payload = "Running Cost %.2f Days \n" % cost
    logger.info(payload)
