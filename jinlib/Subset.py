from torch.utils.data import Dataset

class Subset(Dataset):
  '''
  A custom implementation of torch.utils.data.Subset which supports
  transform specification.

  Arguments:
      dataset (Dataset): The whole Dataset
      indices (sequence): Indices in the whole set selected for subset
  '''
  def __init__(self, dataset, indices, transform=None, sample_i=0):
    self.dataset = dataset
    self.indices = indices
    self.transform = transform
    self.sample_i = sample_i  # index of sample data in each dataset example

  def __getitem__(self, idx):
    example = list(self.dataset[self.indices[idx]])
    if self.transform:
      example[self.sample_i] = self.transform(example[self.sample_i])
    return example

  def __len__(self):
    return len(self.indices)
