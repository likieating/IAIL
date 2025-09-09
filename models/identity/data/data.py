from torchvision import datasets, transforms
from utils.toolkit import read_annotations
import torch
# from imgaug import augmenters as iaa

### 一个空的类，如果要使用的话，就clss xxx(iData)，然后里面定义iData下面的具体数值：类似转换操作
class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None

#这个就是继承了上面那个类
class OSMA(iData):
    use_path = True
    train_trsf = []
    test_trsf = []
    #所有数据集的转换操作，
    common_trsf = [
        # transforms.Grayscale(),
        # transforms.Resize((224,224)),
        # transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        #归一化操作，将像素映射到0.5，0.5，0.5范围里
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    #定义类别顺序，按这个顺序进行训练
    # class_order = [0, 1, 2, 3, 4, 5, 6, 7]

    class_order = [0, 1, 2, 3]

    def process_data(self, train_data_path, val_data_path, test_data_path, out_data_path, debug):
        #得到数据位置和数据的标签
        self.train_data, self.train_targets = read_annotations(train_data_path, debug)
        self.val_data, self.val_targets = read_annotations(val_data_path, debug)
        self.test_data, self.test_targets = read_annotations(test_data_path, debug)
        self.out_data, self.out_targets = read_annotations(out_data_path, debug)

class MultiRandomCrop(torch.nn.Module):
    def __init__(self, num_crop=5, size=299, factor=2):
        super().__init__()
        self.num_crop = num_crop
        self.size = size
        self.factor = factor

    def forward(self, image):
        cropper = transforms.RandomResizedCrop(
            self.size // self.factor,
            ratio=(1, 1),
            antialias=True,
        )
        patches = []
        for _ in range(self.num_crop):
            patches.append(cropper(image))
        return torch.stack(patches, 0)

    def __repr__(self) -> str:
        detail = f"(num_crop={self.num_crop}, size={self.size})"
        return f"{self.__class__.__name__}{detail}"



class ccccc(iData):
    use_path = True
    train_trsf = []
    test_trsf = []
    #所有数据集的转换操作，
    common_trsf = [
        # transforms.Grayscale(),
        # transforms.Resize((224,224)),
        # transforms.CenterCrop((256,256)),
        transforms.ToTensor(),
        MultiRandomCrop(
            num_crop=10, size=299, factor=2
        ),
        #归一化操作，将像素映射到0.5，0.5，0.5范围里
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    #定义类别顺序，按这个顺序进行训练
    # class_order = [0, 1, 2, 3, 4, 5, 6, 7]

    class_order = [0, 1,2,3]

    def process_data(self, train_data_path, val_data_path, test_data_path, out_data_path, debug):
        #得到数据位置和数据的标签
        self.train_data, self.train_targets = read_annotations(train_data_path, debug)
        self.val_data, self.val_targets = read_annotations(val_data_path, debug)
        self.test_data, self.test_targets = read_annotations(test_data_path, debug)
        self.out_data, self.out_targets = read_annotations(out_data_path, debug)


class Wavelet(iData):
    use_path = True
    train_trsf = []
    test_trsf = []
