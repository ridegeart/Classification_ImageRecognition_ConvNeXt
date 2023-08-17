from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms

class MyDataset(Dataset):
    def __init__(self, data_dir, phase,target_names):
        self.transform_train = transforms.Compose([
                        transforms.RandomResizedCrop((224, 224), scale=(0.75, 1.0)),
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
                        transforms.RandomRotation(15),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])
                                            
        self.transform_test = transforms.Compose([
                        transforms.Resize((224, 224), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])])

        self.target1_name = target_names['name1']
        self.target2_name = target_names['name2']
        self.target3_name = target_names['name3']
        self.phase = phase
        if self.phase == 'pred':
            self.image_path = [os.path.join(data_dir, i) for i in os.listdir(data_dir)]
        else:
            self.image_path = []
            self.target1 = []
            self.target2 = []
            self.target3 = []
            class_list = os.listdir(data_dir)
            for i, class_ in enumerate(class_list):
                name1, name2, name3 = class_.split('@')[2], class_.split('@')[1], class_.split('@')[0]
#                if name1 not in self.target1_name:
#                    self.target1_name.append(name1)
#                if name2 not in self.target2_name:
#                    self.target2_name.append(name2)
#                if name3 not in self.target3_name:
#                    self.target3_name.append(name3)
                    
                data_list = [os.path.join(data_dir, class_, i) for i in os.listdir(os.path.join(data_dir, class_))]
                self.image_path.extend(data_list)
                self.target1.extend([self.target1_name.index(name1)]*len(data_list))
                self.target2.extend([self.target2_name.index(name2)]*len(data_list))
                self.target3.extend([self.target3_name.index(name3)]*len(data_list))

    def get_target_name(self):
        return self.target1_name, self.target2_name, self.target3_name

    def __len__(self):
        return len(self.image_path)

    def __getitem__(self,index):
        if self.phase == 'pred':
            img_path= self.image_path[index]
            img = Image.open(img_path).convert('RGB')
            img = self.transform_test(img)
            return img,img_path
        else:
            img_path, label1, label2, label3 = self.image_path[index], self.target1[index], self.target2[index], self.target3[index]
            img = Image.open(img_path).convert('RGB')
            if self.phase == 'train':
                img = self.transform_train(img)
            elif self.phase == 'test':
                img = self.transform_test(img)

            return img, label1, label2, label3
