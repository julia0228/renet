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
from models.dataloader.samplers import CategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.renet import RENet
from models.support_aug import build_support_augmentation
from test import test_main, evaluate


def train(epoch, model, loader, optimizer, args=None):
    model.train()

    train_loader = loader['train_loader']
    train_loader_aux = loader['train_loader_aux']

    # label for query set, always in the same pattern
    label = torch.arange(args.way).repeat(args.query).cuda()  # 012340123401234...

    loss_meter = Meter()
    acc_meter = Meter()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(train_loader)
    pseudo_correct_sum = 0.0
    pseudo_count = 0

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

        use_support_aug = args.use_support_aug and epoch > args.pretrain_epoch
        label_used = label  # 默认所有 query 都参与精度计算

        if use_support_aug:
            with torch.no_grad():
                logits_first, _ = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
                probs = F.softmax(logits_first, dim=-1)
            data_shot_aug, support_weights, pseudo_indices = build_support_augmentation(
                data_shot, data_query, probs, args
            )
            support_weights = support_weights.to(data_shot.device)

            # 将被选为 pseudo support 的 query 从第二次 CCA 的 query 集合中移除
            num_qry = data_query.size(0)
            mask_keep = torch.ones(num_qry, dtype=torch.bool, device=data_query.device)
            pseudo_flat = pseudo_indices.view(-1)
            mask_keep[pseudo_flat] = False
            data_query_second = data_query[mask_keep]

            # 第二次 CCA 只对剩余 query 分类
            logits, absolute_logits = model((data_shot_aug.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1),
                                             data_query_second, support_weights))

            # pseudo query 使用第一次前向的预测参与额外的监督损失
            pseudo_logits = logits_first[pseudo_flat]
            pseudo_labels = label[pseudo_flat]

            # 统计本 epoch 所有 pseudo 的总体正确率（基于第一次前向）
            way = args.way
            topk = max(1, getattr(args, 'pseudo_topk', 1))
            for c in range(way):
                idx_c = pseudo_indices[c].view(-1)
                local_labels = label[idx_c]
                pseudo_correct_sum += (local_labels == c).float().sum().item()                
                pseudo_count += idx_c.numel()
        else:
            logits, absolute_logits = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_query))
            pseudo_logits, pseudo_labels = None, None

        # episodic loss：剩余 query 用第二次 CCA，pseudo query 用第一次预测
        if use_support_aug:
            label_second = label[mask_keep]
            epi_loss_main = F.cross_entropy(logits, label_second)
            epi_loss_pseudo = F.cross_entropy(pseudo_logits, pseudo_labels)
            epi_loss = epi_loss_main + epi_loss_pseudo
            label_used = label_second

            # absolute head 只对第二次 CCA 中的 query 计算
            train_labels_q = train_labels[k:]
            absolute_loss = F.cross_entropy(absolute_logits, train_labels_q[mask_keep])
        else:
            epi_loss = F.cross_entropy(logits, label)
            absolute_loss = F.cross_entropy(absolute_logits, train_labels[k:])

        # loss for auxiliary batch
        model.module.mode = 'fc'
        logits_aux = model(data_aux)
        loss_aux = F.cross_entropy(logits_aux, train_labels_aux)
        loss_aux = loss_aux + absolute_loss

        loss = args.lamb * epi_loss + loss_aux
        acc = compute_accuracy(logits, label_used)

        loss_meter.update(loss.item())
        acc_meter.update(acc)
        phase = 'pseudo' if use_support_aug else 'pretrain'
        tqdm_gen.set_description(f'[train-{phase}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{acc_meter.avg():.3f} (curr:{acc:.3f})')

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

    trainset = Dataset('train', args)
    train_sampler = CategoriesSampler(trainset.label, len(trainset.data) // args.batch, args.way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler, num_workers=8, pin_memory=True)

    trainset_aux = Dataset('train', args)
    train_loader_aux = DataLoader(dataset=trainset_aux, batch_size=args.batch, shuffle=True, num_workers=8, pin_memory=True)

    train_loaders = {'train_loader': train_loader, 'train_loader_aux': train_loader_aux}

    valset = Dataset('val', args)
    val_sampler = CategoriesSampler(valset.label, args.val_episode, args.way, args.shot + args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler, num_workers=8, pin_memory=True)
    ''' fix val set for all epochs '''
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

    for epoch in range(start_epoch, args.max_epoch + 1):
        start_time = time.time()

        train_loss, train_acc, _ = train(epoch, model, train_loaders, optimizer, args)

        if epoch > args.pretrain_epoch:
            val_loss, val_acc, _ = evaluate(epoch, model, val_loader, args, set='val')

            if not args.no_wandb:
                wandb.log({'train/loss': train_loss, 'train/acc': train_acc,
                           'val/loss': val_loss, 'val/acc': val_acc}, step=epoch)

            if val_acc > max_acc:
                print(f'[ log ] *********A better model is found ({val_acc:.3f}) *********')
                max_acc, max_epoch = val_acc, epoch
                torch.save(dict(params=model.state_dict(), epoch=epoch), os.path.join(args.save_path, 'max_acc.pth'))
                torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_max_acc.pth'))
        else:
            # 预训练阶段只记录训练指标
            if not args.no_wandb:
                wandb.log({'train/loss': train_loss, 'train/acc': train_acc}, step=epoch)

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
