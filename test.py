# test with pseudo
import os
import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from common.meter import Meter
from common.utils import compute_accuracy, load_model, setup_run, by
from models.dataloader.samplers import CategoriesSampler, PairedCategoriesSampler
from models.dataloader.data_utils import dataset_builder
from models.dataloader.pair_dataset import PairedDataset
from models.renet import RENet
from models.support_aug import build_support_augmentation


def evaluate(epoch, model, loader, args=None, set='val'):
    model.eval()

    loss_meter = Meter()
    acc_meter = Meter()

    label = torch.arange(args.way).repeat(args.query).cuda()

    k = args.way * args.shot
    tqdm_gen = tqdm.tqdm(loader)

    with torch.no_grad():
        for i, batch in enumerate(tqdm_gen, 1):
            # 解析批次数据 - 新的数据格式
            (data, labels), unlabeled_data = batch
            
            data = data.cuda()
            unlabeled_data = unlabeled_data.cuda()  # 无标签数据也需要移动到GPU

            # 移除多余的批次维度
            if data.dim() == 5 and data.size(0) == 1:
                data = data.squeeze(0)
            if labels.dim() > 1 and labels.size(0) == 1:
                labels = labels.squeeze(0)
            if unlabeled_data.dim() == 5 and unlabeled_data.size(0) == 1:
                unlabeled_data = unlabeled_data.squeeze(0)
            
            model.module.mode = 'encoder'
            data = model(data)
            # 如果需要，也可以对无标签数据进行编码
            data_un = model(unlabeled_data)
            
            data_shot, data_query = data[:k], data[k:]
            model.module.mode = 'cca'

            logits_first = model((data_shot.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1), data_un))
            probs = F.softmax(logits_first, dim=-1)
            data_shot_aug, support_weights, _ = build_support_augmentation(
                    data_shot, data_query, probs, args
            )
            support_weights = support_weights.to(data_shot.device)

            logits = model((data_shot_aug.unsqueeze(0).repeat(args.num_gpu, 1, 1, 1, 1),
                                data_query, support_weights))

            loss = F.cross_entropy(logits, label)
            acc = compute_accuracy(logits, label)

            loss_meter.update(loss.item())
            acc_meter.update(acc)
            tqdm_gen.set_description(f'[{set:^5}] epo:{epoch:>3} | avg.loss:{loss_meter.avg():.4f} | avg.acc:{by(acc_meter.avg())} (curr:{acc:.3f})')

    return loss_meter.avg(), acc_meter.avg(), acc_meter.confidence_interval()


def test_main(model, args):

    ''' load model '''
    model = load_model(model, os.path.join(args.save_path, 'max_acc.pth'))

    ''' define test dataset '''
    Dataset = dataset_builder(args)
    # 创建测试集的有标签和无标签数据集
    testset_labeled = Dataset('test', args)
    testset_unlabeled = Dataset('test_unlabeled', args)  # 假设你有测试集的无标签数据

    # 创建测试集的配对采样器
    test_sampler = PairedCategoriesSampler(
        labeled_label=testset_labeled.label,
        unlabeled_label=testset_unlabeled.label,
        n_batch=args.test_episode,
        n_cls=args.way,
        n_per_labeled=args.shot + args.query,
        n_per_unlabeled=args.unlabeled
    )

    # 创建测试集的配对数据集
    paired_testset = PairedDataset(testset_labeled, testset_unlabeled, test_sampler)

    # 创建测试集的数据加载器
    test_loader = DataLoader(
        dataset=paired_testset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        pin_memory=True
    )

    ''' evaluate the model with the dataset '''
    _, test_acc, test_ci = evaluate("best", model, test_loader, args, set='test')
    print(f'[final] epo:{"best":>3} | {by(test_acc)} +- {test_ci:.3f}')

    return test_acc, test_ci


if __name__ == '__main__':
    args = setup_run(arg_mode='test')

    ''' define model '''
    model = RENet(args).cuda()
    model = nn.DataParallel(model, device_ids=args.device_ids)

    test_main(model, args)
