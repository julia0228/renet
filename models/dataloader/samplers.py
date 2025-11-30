import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch  # the number of iterations in the dataloader
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)  # all data label
        self.m_ind = []  # the data index of each class
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)  # all data index of this class
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]  # random sample num_class indices, e.g. 5
            for c in classes:
                l = self.m_ind[c]  # all data indices of this class
                pos = torch.randperm(len(l))[:self.n_per]  # sample n_per data index of this class
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            # .t() transpose,
            # due to it, the label is in the sequence of abcdabcdabcd form after reshape,
            # instead of aaaabbbbccccdddd
            yield batch

class PairedCategoriesSampler():

    def __init__(self, labeled_label, unlabeled_label, n_batch, n_cls, n_per_labeled, n_per_unlabeled):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per_labeled = n_per_labeled
        self.n_per_unlabeled = n_per_unlabeled

        # 分别构建两个数据集的索引
        labeled_label = np.array(labeled_label)
        unlabeled_label = np.array(unlabeled_label)
        
        self.labeled_m_ind = []
        self.unlabeled_m_ind = []
        
        for i in range(max(max(labeled_label), max(unlabeled_label)) + 1):
            # 有标签数据的索引
            labeled_ind = np.argwhere(labeled_label == i).reshape(-1)
            self.labeled_m_ind.append(torch.from_numpy(labeled_ind))
            
            # 无标签数据的索引
            unlabeled_ind = np.argwhere(unlabeled_label == i).reshape(-1)
            self.unlabeled_m_ind.append(torch.from_numpy(unlabeled_ind))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch_labeled = []
            batch_unlabeled = []
            
            # 随机选择相同的类别集合
            classes = torch.randperm(len(self.labeled_m_ind))[:self.n_cls]
            
            for c in classes:
                # 从有标签数据中采样
                l_labeled = self.labeled_m_ind[c]
                pos_labeled = torch.randperm(len(l_labeled))[:self.n_per_labeled]
                batch_labeled.append(l_labeled[pos_labeled])
                
                # 从无标签数据中采样（相同类别）
                l_unlabeled = self.unlabeled_m_ind[c]
                pos_unlabeled = torch.randperm(len(l_unlabeled))[:self.n_per_unlabeled]
                batch_unlabeled.append(l_unlabeled[pos_unlabeled])
            
            # 分别处理两个batch
            batch_labeled = torch.stack(batch_labeled).t().reshape(-1)
            batch_unlabeled = torch.stack(batch_unlabeled).t().reshape(-1)
            
            yield batch_labeled, batch_unlabeled
