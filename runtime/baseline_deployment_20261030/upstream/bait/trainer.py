import time
import models
import torch
import util
import toolbox

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Trainer():
    def __init__(self, criterion, data_loader, logger, config, global_step=0,
                 target='train_dataset'):
        self.criterion = criterion
        self.data_loader = data_loader
        self.logger = logger
        self.config = config
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.global_step = global_step
        self.target = target
        print(self.target)

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()

    def train(self, args, epoch, model, atk, criterion, optimizer, meta_train=False):
        model.train()
        for i, (images, labels) in enumerate(self.data_loader[self.target]):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)

            start = time.time()
            log_payload = self.train_batch(images, labels, model, optimizer, meta_train)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                self.logger.info(display)
            self.global_step += 1
        return self.global_step

    def train_batch(self, images, labels, model, optimizer, meta_train=False):
        if meta_train:
            if isinstance(self.criterion, torch.nn.CrossEntropyLoss) or isinstance(self.criterion, models.CutMixCrossEntropyLoss):
                logits = model(images)
                loss = self.criterion(logits, labels) 
            optimizer.step(loss)  
        else:
            model.zero_grad()  
            optimizer.zero_grad()
            if isinstance(self.criterion, torch.nn.CrossEntropyLoss) or isinstance(self.criterion, models.CutMixCrossEntropyLoss):
                logits = model(images)
                loss = self.criterion(logits, labels) 
            loss.backward()  
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
            optimizer.step()  
        if logits.shape[1] >= 5:
            acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
            acc, acc5 = acc.item(), acc5.item()
        else:
            acc, = util.accuracy(logits, labels, topk=(1,))
            acc, acc5 = acc.item(), 1
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(acc, labels.shape[0])
        self.acc5_meters.update(acc5, labels.shape[0])
        payload = {"acc": acc,
                "acc_avg": self.acc_meters.avg,
                "loss": loss,
                "loss_avg": self.loss_meters.avg,
                "lr": optimizer.param_groups[0]['lr']}
        return payload

