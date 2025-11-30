# train with pseudo
import os
import tqdm
import time
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import detect_grad_nan, compute_accuracy, set_seed, setup_run
from models.dataloader.samplers import CategoriesSampler, PairedCategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.dataloader.pair_dataset import PairedDataset
from models.renet import RENet
from models.support_aug import build_support_augmentation
from test import test_main, evaluate

def pretrain(epoch, model, loader, optimizer, args=None):
    model.train()

    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']

    # label for query set, always in the same pattern
    label = torch.arange(args.way).repeat(args.query).cuda()  # 012340123401234...

    loss_meter = Meter()
    acc_meter = Meter()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(train_loader)

    for i, ((data, train_labels), (data_aux, train_labels_aux)) in enumerate(zip(tqdm_gen, train_loader_aux), 1):

        data, train_labels = data.cuda(), train_labels.cuda()
        data_aux, train_labels_aux = data_aux.cuda(), train_labels_aux.cuda()

        # Forward images (3, 84, 84) -> (C, H, W)
        model.module.mode = 'encoder'
        data = model(data)
        data_aux = model(data_aux)  # I prefer to separate feed-forwarding data and data_aux due to BN

        # loss for batch
        model.module.mode = 'cca'
        data_shot, data_query = data[:k], data[k:]
        logits, absolute_logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
        epi_loss = F.cross_entropy(logits, label)
        absolute_loss = F.cross_entropy(absolute_logits, train_labels[k:])

        # loss for auxiliary batch
        model.module.mode = 'fc'
        logits_aux = model(data_aux)
        loss_aux = F.cross_entropy(logits_aux, train_labels_aux)
        loss_aux = loss_aux + absolute_loss

        loss = args.lamb * epi_loss + loss_aux
        acc = compute_accuracy(logits, label)

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[pretrain] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()

def train(epoch, model, loader, optimizer, args=None):
    model.train()

    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']

    # label for query set, always in the same pattern
    label = torch.arange(args.way).repeat(args.query).cuda()  # 012340123401234...
    un_label = torch.arange(args.way).repeat(args.unlabeled).cuda()

    loss_meter = Meter()
    acc_meter = Meter()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(zip(train_loader, train_loader_aux), 
                         total=len(train_loader), 
                         desc='Training')

    pseudo_correct_sum = 0.0
    pseudo_count = 0

    for i, (batch, aux_batch) in enumerate(tqdm_gen, 1):
        # 解析主批次数据
        (data, train_labels), unlabeled_data = batch
        data, train_labels = data.cuda(), train_labels.cuda()
        unlabeled_data = unlabeled_data.cuda()

        # 移除多余的批次维度
        if data.dim() == 5 and data.size(0) == 1:
            data = data.squeeze(0)
        if train_labels.dim() > 1 and train_labels.size(0) == 1:
            train_labels = train_labels.squeeze(0)
        if unlabeled_data.dim() == 5 and unlabeled_data.size(0) == 1:
            unlabeled_data = unlabeled_data.squeeze(0)
        
        # 解析辅助批次数据
        data_aux, train_labels_aux = aux_batch
        data_aux, train_labels_aux = data_aux.cuda(), train_labels_aux.cuda()

        # Forward images (3, 84, 84) -> (C, H, W)
        model.module.mode = 'encoder'
        data = model(data)
        data_aux = model(data_aux)  # I prefer to separate feed-forwarding data and data_aux due to BN
        data_un = model(unlabeled_data)

        # loss for batch
        model.module.mode = 'cca'
        data_shot, data_query = data[:k], data[k:]

        with torch.no_grad():
                logits_first, _ = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_un))
                probs = F.softmax(logits_first, dim=-1)
        data_shot_aug, support_weights, pseudo_indices = build_support_augmentation(
            data_shot, data_un, probs, args
        )
        support_weights = support_weights.to(data_shot.device)

        # pseudo_flat = pseudo_indices.view(-1)

        # absolute_logits 是 RENet 中“绝对分类头”的输出，对应于标准的有监督分类 logits（类别数 = 训练集总类别数），
        # 与 episodic 的 few-shot logits 不同，后者只在当前 episode 的 way 个类别上做分类。
        logits, absolute_logits = model((data_shot_aug.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1),
                                             data_query, support_weights))

        # # pseudo query 使用第一次前向的预测参与额外的监督损失
        # pseudo_logits = logits_first[pseudo_flat]
        # pseudo_labels = label[pseudo_flat]

        # 统计本 epoch 所有 pseudo 的总体正确率（基于第一次前向）
        way = args.way
        topk = max(1, getattr(args, 'pseudo_topk', 1))
        for c in range(way):
            idx_c = pseudo_indices[c].view(-1)
            local_labels = un_label[idx_c]
            pseudo_correct_sum += (local_labels == c).float().sum().item()                
            pseudo_count += idx_c.numel()

        # episodic loss
        epi_loss = F.cross_entropy(logits, label)

        # absolute head
        absolute_loss = F.cross_entropy(absolute_logits, train_labels[k:])

        # loss for auxiliary batch
        model.module.mode = 'fc'
        logits_aux = model(data_aux)
        loss_aux = F.cross_entropy(logits_aux, train_labels_aux)
        loss_aux = loss_aux + absolute_loss

        loss = args.lamb * epi_loss + loss_aux
        acc = compute_accuracy(logits, label)

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        tqdm_gen.set_description(f'[train] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        detect_grad_nan(model)
        optimizer.step()
        optimizer.zero_grad()

    if pseudo_count > 0:
        avg_pseudo_acc = pseudo_correct_sum / pseudo_count
        print(f'[pseudo-epoch] epo:{epoch} pseudo_acc:{avg_pseudo_acc:.3f} '
              f'(over {int(pseudo_count)} pseudo supports)')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def train_main(args):
    Dataset = dataset_builder(args)

    # 创建两个数据集
    trainset_labeled = Dataset('train', args)
    trainset_unlabeled = Dataset('train_unlabeled', args)

    # Pretrain sampler & loader
    pretrain_sampler = CategoriesSampler(
        trainset_labeled.label,
        n_batch=len(trainset_labeled.data) // args.batch,
        n_cls=args.way,
        n_per=args.shot + args.query
    )
    pretrain_loader = DataLoader(dataset=trainset_labeled, batch_sampler=pretrain_sampler, 
                                num_workers=8, pin_memory=True)

    # 使用新的配对采样器
    train_sampler = PairedCategoriesSampler(
        labeled_label=trainset_labeled.label,
        unlabeled_label=trainset_unlabeled.label, 
        n_batch=len(trainset_labeled.data) // args.batch,
        n_cls=args.way,
        n_per_labeled=args.shot + args.query,  # 保持原来的 shot + query
        n_per_unlabeled=args.unlabeled 
    )
    train_paired_dataset = PairedDataset(trainset_labeled, trainset_unlabeled, train_sampler)
    train_loader = DataLoader(dataset=train_paired_dataset, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)

    # 原有的辅助数据加载器保持不变
    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)

    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}

    # 创建验证集的有标签和无标签数据集
    valset_labeled = Dataset('val', args)
    valset_unlabeled = Dataset('val_unlabeled', args)  # 假设你有验证集的无标签数据

    # 创建验证集的配对采样器
    val_sampler = PairedCategoriesSampler(
        labeled_label=valset_labeled.label,
        unlabeled_label=valset_unlabeled.label,
        n_batch=args.val_episode,
        n_cls=args.way,
        n_per_labeled=args.shot + args.query,
        n_per_unlabeled=args.unlabeled
    )

    # 创建验证集的配对数据集
    paired_valset = PairedDataset(valset_labeled, valset_unlabeled, val_sampler)

    # 创建验证集的数据加载器
    val_loader = DataLoader(
        dataset=paired_valset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    # 固定验证集用于所有epoch
    val_loader = [x for x in val_loader]

    set_seed(args.seed)
    model = RENet(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    start_epoch = 1
    if getattr(args, 'resume_path', ''):
        checkpoint = torch.load(args.resume_path)
        model.load_state_dict(checkpoint['params'])
        start_epoch = 61
        print(f'[ log ] resumed model from {args.resume_path} at epoch {checkpoint.get("epoch", "unknown")}')

    if not args.no_wandb:
        wandb.watch(model)
    print(model)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, nesterov=True, weight_decay=0.0005)
    if getattr(args, 'resume_opt_path', ''):
        opt_state = torch.load(args.resume_opt_path)
        optimizer.load_state_dict(opt_state)
        print(f'[ log ] resumed optimizer from {args.resume_opt_path}')
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    lr_scheduler.last_epoch = start_epoch - 1

    max_acc, max_epoch = 0.0, 0
    set_seed(args.seed)

    # 预训练阶段
    pretrain_epochs = getattr(args, 'pretrain_epoch', 0)
    if pretrain_epochs > 0:
        print(f'[ log ] Starting pretraining for {pretrain_epochs} epochs')
        pretrain_loaders = {
            'train_loader': pretrain_loader,
            'train_loader_aux': train_loader_aux
        }
        
        for epoch in range(start_epoch, start_epoch + pretrain_epochs):
            start_time = time.time()
            
            train_loss, train_acc, _ = pretrain(epoch, model, pretrain_loaders, optimizer, args)
            
            if not args.no_wandb:
                wandb.log({'pretrain/loss': train_loss, 'pretrain/acc': train_acc}, step=epoch)
            
            epoch_time = time.time() - start_time
            print(f'[ log ] pretrain epoch {epoch} completed in {epoch_time:.2f}s')
            
            lr_scheduler.step()
        
        start_epoch += pretrain_epochs
        print(f'[ log ] Pretraining completed, starting normal training from epoch {start_epoch}')
    
    # 正常训练阶段
    for epoch in range(start_epoch, args.max_epoch + 1):
        start_time = time.time()

        train_loss, train_acc, _ = train(epoch, model, train_loaders, optimizer, args)

        val_loss, val_acc, _ = evaluate(epoch, model, val_loader, args, set='val')

        if not args.no_wandb:
            wandb.log({'train/loss': train_loss, 'train/acc': train_acc,
                           'val/loss': val_loss, 'val/acc': val_acc}, step=epoch)

        if val_acc > max_acc:
            print(f'[ log ] *********A better model is found ({val_acc:.3f}) *********')
            max_acc, max_epoch = val_acc, epoch
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_max_acc.pth'))

        if args.save_all:
            torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, f'epoch_{epoch}.pth'))
            torch.save(optimizer.state_dict(), os.path.join(args.save_path, f'optimizer_epoch_{epoch}.pth'))

        epoch_time = time.time() - start_time
        print(f'[ log ] saving @ {args.save_path}')
        print(f'[ log ] roughly {(args.max_epoch - epoch) / 3600. * epoch_time:.2f} h left\n')

        lr_scheduler.step()

    return model


if __name__ == '__main__':
    args = setup_run(arg_mode='train')

    model = train_main(args)
    test_acc, test_ci = test_main(model, args)

    if not args.no_wandb:
        wandb.log({'test/acc': test_acc, 'test/confidence_interval': test_ci})
