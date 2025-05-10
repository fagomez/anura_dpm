from torch.utils.data import Dataset
import os


class GenerativeAIDataset(Dataset):
    '''
    Dataset for the generative AI models

    Args:
    folder (str): The folder with the data
    transform (Compose): The data composing
    class_c (str): The class to use
    expand_data (int): The number of times to expand the data
    '''
    def __init__(self, folder, transform=None, class_c=None, expand_data=1) -> None:
        all_classes = [
            d for d in os.listdir(folder)\
                if os.path.isdir(os.path.join(folder, d)) and not d.startswith('_')]
        
        data = []

        for c in all_classes:
            d = os.path.join(folder, c)
            if class_c is not None and c != class_c:
                continue
            for f in os.listdir(d):
                path = os.path.join(d, f)
                data.append(path)

        self.data = data
        self.transform = transform
        self.expand_data = expand_data

    def __len__(self):
        return len(self.data) * self.expand_data

    def __getitem__(self, index):
        real_index = index % len(self.data)
        path = self.data[real_index]
        data = {'path': path}

        if self.transform is not None:
            data = self.transform(data)

        return data
    