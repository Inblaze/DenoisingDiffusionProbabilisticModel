from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image

class MyDataset(Dataset):
    def __init__(self,
                 root:str,
                 transform=None):
        self.root = root
        self.transform = transform
        self.images, self.labels = self._get_images_and_labels(self.root)
 
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = self.images[index]
        labels = self.labels[index]
        imgs = Image.open(img_path)
        imgs.load()
        imgs = imgs.convert('RGB')
        if self.transform:
            imgs = self.transform(imgs)
        return imgs, labels
    
    @staticmethod
    def _get_images_and_labels(root:str, num_sub_cls:int=11) -> tuple[list, list]:
        '''
        :param root: root of the dataset
        :return: images_list, labels_list
        '''
        root = Path(root)

        sex_classes = []
        age_classes = []

        for i in root.iterdir():
            if i.is_dir():
                sex_classes.append(i.name)
                for j in i.iterdir():
                    if j.is_dir():
                        if j.name not in age_classes:
                            age_classes.append(j.name)
    
        images_list = [] 
        labels_list = []
 
        for sex_i, sex in enumerate(sex_classes):
            sex_path = root / sex
            if not sex_path.is_dir():
                continue
            for age_i, age in enumerate(age_classes):
                class_path = sex_path / age
                if not class_path.is_dir():
                    continue
                for img_path in class_path.glob('*.jpg'):
                    images_list.append(str(img_path))
                    labels_list.append(int(sex_i * num_sub_cls + age_i))

        return images_list, labels_list
