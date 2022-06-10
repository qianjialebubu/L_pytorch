import os
from PIL import Image
from torch.utils.data import Dataset


class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir,label_dir)
        self.image_path = os.listdir(self.path)

    def __getitem__(self,idx):
        img_name = self.image_path
        img_item_path = os.path.join(self.root_dir,self.label_dir,img_name[idx])
        img = Image.open(img_item_path)
        label = self.label_dir
        return img,label

    def __len__(self):
        return len(self.image_path)

root_dir = "D:/file/work_space/deep_leaning/L_pytorch/hymenoptera_data/train"
label_dir_ants = "ants"
label_dir_bees = "bees"
ants_MyData = MyData(root_dir, label_dir_ants)
bees_MyData = MyData(root_dir, label_dir_bees)
MyData = ants_MyData+bees_MyData
print(MyData.__len__())

img , label = ants_MyData.__getitem__(2)
img.show()
# print(ants_MyData.__getitem__(0))
# img,label = ants_MyData[0]
# img.show()