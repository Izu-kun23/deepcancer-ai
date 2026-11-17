from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset

MEAN = [.485, .456, .406]
STD = [.229, .224, .225]

def transform(training=False):
    ops = [transforms.Resize((224, 224))]
    if training:
        ops += [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(), transforms.RandomRotation(15)]
    return transforms.Compose(ops + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

class TissueDataset(Dataset):
    def __init__(self, frame, training=False):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform(training)
    def __len__(self):
        return len(self.frame)
    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(row.path) as im:
            x = self.transform(im.convert('RGB'))
        return x, int(row.label)
