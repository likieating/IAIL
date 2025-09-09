import logging
import numpy as np
import torch
from PIL import Image, ImageFile, ImageFilter
import random
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset
from torchvision import transforms
from utils.data import OSMA
from utils.data import ccccc
#
class DataManager(object):
    #初始化数据管理器，配置和准备数据集
    def __init__(self, dataset_name, train_data_path, val_data_path, test_data_path, out_data_path, shuffle, seed, init_cls, increment,
                 debug):
        self.get_first = 0
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, train_data_path, val_data_path, test_data_path, out_data_path, shuffle, seed, debug)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        self.memory_samples=[]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    def get_all_train_samples(self):
        """
        获取所有已学习的训练样本。

        返回:
        - all_samples: Sample 对象的列表
        """
        return self.memory_samples.copy()

   #得到增量学习的序列
    @property
    def nb_tasks(self):
        return len(self._increments)

    #获取指定任务阶段的类别数量
    def get_task_size(self, task):
        return self._increments[task]

    #获取总类别数量
    def get_total_classnum(self):
        return len(self._class_order)


    #将数据加载进来之后，将数据存储入变量，并返回data,targets，和封装了这些数据的dummydataset
    def get_dataset(
            self, indices, source, mode, resize_size=128, appendent=None, distinguish=False,ret_data=False, m_rate=None,indicies_direct=None
    ):
        if source == "train":
            x, y = self._train_data, self._train_targets
            if not self.get_first:
                class_data_len = []
                for i in np.arange(len(np.unique(y))):
                    class_data_len.append((i, len(x[y == i])))
                logging.info("the length of each class(total) in train: {}".format(class_data_len))
                self.get_first += 1
        elif source == "val":
            x, y = self._val_data, self._val_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        elif source == "test_out":
            x, y = self._out_data, self._out_targets
            indices = np.unique(y)
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transforms.Compose(
                [
                    *self._test_trsf,
                    transforms.RandomHorizontalFlip(p=1.0),
                    *self._common_trsf,
                ]
            )
        elif mode == "val" or mode == "test":
            trsf = transforms.Compose([*self._test_trsf, *self._common_trsf])
        elif mode=="random_choose":
            trsf = transforms.Compose([*self._train_trsf, *self._common_trsf])
            data,targets=[],[]

            for idx in indices:
                for i in range(100):
                    data.append(x[idx])
                    targets.append(x[idx])
            data, targets = np.concatenate(data), np.concatenate(targets)
            if ret_data:
                return data, targets, DummyDataset(self.dataset_name, data, targets, resize_size, trsf, mode,
                                                   self.use_path)
            else:
                # return DummyDataset(self.dataset_name, data, targets, resize_size, trsf, self.use_path, balance=True)
                return DummyDataset(self.dataset_name, data, targets, resize_size, trsf, mode, self.use_path)
        else:
            raise ValueError("Unknown mode {}.".format(mode))
        data, targets = [], []
        for idx in indices:
            data.append(x[y == idx])
            targets.append(y[y == idx])
        if indicies_direct is not None:
            for idx in indicies_direct:
                data.append(x[idx])
                targets.append(x[idx])

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(self.dataset_name, data, targets, resize_size, trsf, mode, self.use_path)
        elif distinguish:
            return DummyDataset(self.dataset_name, data, targets, resize_size, trsf, mode, self.use_path,balance=True)
        else:
            # return DummyDataset(self.dataset_name, data, targets, resize_size, trsf, self.use_path, balance=True)
            return DummyDataset(self.dataset_name, data, targets, resize_size, trsf, mode, self.use_path)

#进行一些数据在进入模型训练之前的通用预处理：必要的变换
    def _setup_data(self, dataset_name, train_data_path, val_data_path, test_data_path, out_data_path, shuffle, seed, debug):
        #拿到数据并对数据进行基本处理（归一化）
        idata = _get_idata(dataset_name)
        # 拿到数据集中的所有data和对应的targets
        idata.process_data(train_data_path, val_data_path, test_data_path, out_data_path, debug)

        # 存储数据和标签
        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._val_data, self._val_targets = idata.val_data, idata.val_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self._out_data, self._out_targets = idata.out_data, idata.out_targets
        self.use_path = idata.use_path

        # Transforms
        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf
        self.abu_trsf=idata.albumentations_transform

        # Order：如果shuffle=true,那么就把类别顺序随机打乱
        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info("the random class order of data: {}".format(self._class_order))

        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._val_targets = _map_new_class_index(self._val_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(
                0, len(idxes), size=int((1 - m_rate) * len(idxes))
            )
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))

#将图像路径和类别标签封装在一起，并保持真伪图片之间的类别平衡，resize在这里面
class DummyDataset(Dataset):
    def __init__(self, dataset_name, images, orders, resize_size, trsf, mode, use_path=False, balance=None):
        assert len(images) == len(orders), "Data size error!"
        self.balance = balance
        if balance:
            self.real_images, self.fake_images, self.real_orders, self.fake_orders = [], [], [], []
            for idx, image in enumerate(images):
                if "real" in image:
                    self.real_images.append(image)
                    self.real_orders.append(orders[idx])
                else:
                    self.fake_images.append(image)
                    self.fake_orders.append(orders[idx])
        else:
            self.images = images
            self.orders = orders
        self.resize_size = resize_size
        self.dataset_name = dataset_name
        self.trsf = trsf
        self.use_path = use_path
        self.mode = mode

    def __len__(self):
        if self.balance:
            return len(self.real_images)
        else:
            return len(self.images)

    def get_images_from(self, idx, images):
        if idx >= len(images):
            raise IndexError(f"Index {idx} is out of range. Images list length: {len(images)}")
        path = str(images[idx])
        image = self.trsf(pil_loader(path, self.resize_size))
        return path, image

    def __getitem__(self, idx):
        if self.balance:
            idx=idx%min(len(self.fake_images),len(self.real_images))
            img_paths = []
            fake_path, fake_image = self.get_images_from(idx, self.fake_images)
            real_path, real_image = self.get_images_from(idx, self.real_images)
            img_paths.extend((real_path, fake_path))
            image = torch.cat([real_image.unsqueeze(0), fake_image.unsqueeze(0)], dim=0)
            label = torch.tensor([0,1], dtype=torch.long)
            order = torch.tensor([self.real_orders[idx], self.fake_orders[idx]])
            return img_paths, image, label, order
        else:
            path, image = self.get_images_from(idx, self.images)
            # else:
            #     image = self.trsf(Image.fromarray(self.images[idx]))
            if 'real' in path:
                label = 0
            else:
                label = 1
            order = self.orders[idx]
            return path, image, label, order

#将原始类别标签映射到新的类别索引中
def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))

#输入OSMA就返回OSMA实例（这个实例里面有数据集的位置和处理方法之类的）
def _get_idata(dataset_name):
    name = dataset_name.upper()
    if name == "OSMA" or name == "OSMA_LNP":
        return OSMA()
    elif name=="CCCCC":
        return ccccc()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))

#自定义图像加载函数，读取并进行预处理
def pil_loader(path, resize_size):
    """
    Ref:
    https://pytorch.org/docs/stable/_modules/torchvision/datasets/folder.html#ImageFolder
    """
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    img = Image.open(path).convert("RGB")
    # if img.size[0] != img.size[1]:
    #     short_size = img.size[0] if img.size[0] < img.size[1] else img.size[1]
    #     img = transforms.CenterCrop(size=(short_size, short_size))(img)
    # if resize_size is not None:
    #     resize_size = (resize_size, resize_size)
    #     img = img.resize(resize_size)
    if img.size[0] < resize_size or img.size[1] < resize_size:
        if img.size[0] != img.size[1]:
            short_size = img.size[0] if img.size[0] < img.size[1] else img.size[1]
            img = transforms.CenterCrop(size=(short_size, short_size))(img)
        img = transforms.Resize(resize_size)(img)
    elif resize_size is not None:
        img = transforms.RandomCrop(resize_size)(img)
    # img = img.filter(ImageFilter.Kernel((3, 3), (0, -1, 0, -1, 4, -1, 0, -1, 0)))
    return img.convert("RGB")

