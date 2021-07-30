from torch.utils.data import Dataset

class Subset(Dataset):
  '''
  A custom implementation of torch.utils.data.Subset which supports
  transform specification.

  Arguments:
      dataset (Dataset): The whole Dataset
      indices (sequence): Indices in the whole set selected for subset
  '''
  def __init__(self, dataset, indices, transform=None):
    self.dataset = dataset
    self.indices = indices
    self.transform = transform

  def __getitem__(self, idx):
    example = self.dataset[self.indices[idx]]
    sample = example[0]
    label = example[1]
    sample = self.transform(sample) if self.transform else sample
    return sample, label

  def __len__(self):
    return len(self.indices)
