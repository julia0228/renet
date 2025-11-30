import torch
from torch.utils.data import Dataset, DataLoader

class PairedDataset(Dataset):
    def __init__(self, labeled_dataset, unlabeled_dataset, sampler):
        self.labeled_dataset = labeled_dataset
        self.unlabeled_dataset = unlabeled_dataset
        self.sampler = sampler
        # 预生成所有batch的索引，这样多worker可以安全访问
        self.all_batch_indices = list(sampler)
    
    def __len__(self):
        return len(self.all_batch_indices)
    
    def __getitem__(self, idx):
        labeled_indices, unlabeled_indices = self.all_batch_indices[idx]
        
        # 加载有标签数据
        labeled_data = [self.labeled_dataset[i] for i in labeled_indices]
        labeled_images = torch.stack([item[0] for item in labeled_data])
        labeled_labels = torch.tensor([item[1] for item in labeled_data])
        
        # 加载无标签数据（只取图像，忽略标签）
        unlabeled_data = [self.unlabeled_dataset[i] for i in unlabeled_indices]
        unlabeled_images = torch.stack([item[0] for item in unlabeled_data])
        
        return (labeled_images, labeled_labels), unlabeled_images