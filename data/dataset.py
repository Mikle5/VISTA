r""" Dataloader builder for few-shot semantic segmentation dataset  """
import torch
from torchvision import transforms
from torch.utils.data import DataLoader

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
# from data.coco2pascal import DatasetCOCO2PASCAL


# file: fss_dataset.py

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from data.pascal import DatasetPASCAL
from data.coco import DatasetCOCO
from data.refcocog import DatasetRefcocog  # <-- импортируем новый датасет

class FSSDataset:

    @classmethod
    def initialize(cls, img_size, datapath, use_original_imgsize):
        cls.datasets = {
            'pascal': DatasetPASCAL,
            'coco': DatasetCOCO,
            'refcocog': DatasetRefcocog,  # <-- добавляем сюда
            # 'coco2pascal': DatasetCOCO2PASCAL,
        }

        cls.img_mean = [0.485, 0.456, 0.406]
        cls.img_std  = [0.229, 0.224, 0.225]
        cls.datapath = datapath
        cls.use_original_imgsize = use_original_imgsize

        # Стандартные трансформации (Resize -> ToTensor -> Normalize)
        cls.transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(cls.img_mean, cls.img_std)
        ])


    @classmethod
    def build_dataloader(cls, benchmark, bsz, nworker, fold, split, shot=1):
        """
        benchmark: строка 'pascal', 'coco', 'refcocog' и т.д.
        bsz: batch size
        nworker: num_workers
        fold: номер fold (для k-fold)
        split: 'trn', 'val' или 'tst'
        shot: количество support-примеров
        """
        # Во время обучения используем shuffle для разнообразия эпизодов
        shuffle = (split == 'trn')
        # Для валидации/теста отключаем shuffle и ставим num_workers=0 для воспроизводимости
        nworker = nworker if split == 'trn' else 0

        # Создаём экземпляр датасета
        dataset = cls.datasets[benchmark](
            cls.datapath,
            fold=fold,
            transform=cls.transform,
            split=split,
            shot=shot,
            use_original_imgsize=cls.use_original_imgsize
        )

        # DistributedSampler обычно используется в DDP-тренировке
        sampler = DistributedSampler(dataset, shuffle=shuffle)

        # pin_memory ускоряет передачу на GPU. Обычно включают при использовании CUDA.
        dataloader = DataLoader(
            dataset,
            batch_size=bsz,
            shuffle=False,        # при использовании sampler нужно ставить shuffle=False
            pin_memory=True,
            num_workers=nworker,
            sampler=sampler
        )

        return dataloader
