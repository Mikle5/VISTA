import os
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
import PIL.Image as Image
from PIL import ImageDraw

from torch.utils.data import Dataset
from datasets import load_dataset


class DatasetRefcocog(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        """
        Параметры аналогичны DatasetCOCO:
        :param datapath:            Путь к корневой папке (если нужно, например, для локальных изображений)
        :param fold:                Номер фолда (в RefCOCOG не используется, заглушка)
        :param transform:           torchvision-трансформация (Resize / ToTensor / Normalize) для query
        :param split:               'trn' | 'val' | 'test' — выбираем, какой split грузить
        :param shot:                Количество support-примеров
        :param use_original_imgsize:Если True — не ресайзим изображение/маску, иначе применяем self.transform
        """
        # -- Поля для совместимости с вашим кодом
        self.split = 'validation' if split in ['val', 'test'] else 'train'  # 'val' или 'trn'
        self.fold = fold
        self.nfolds = 1
        self.benchmark = 'refcocog'
        self.shot = shot
        self.transform = transform
        self.use_original_imgsize = use_original_imgsize

        self.base_path = datapath  # Если нужно для дополнительных путей (может не использоваться)

        # -- Определяем, какие сплиты брать из датасета jxu124/refcocog
        
        # -- Загружаем датасет
        self.dataset = load_dataset("jxu124/refcocog", split=self.split)
        
        # -- Построим группировку по category_id, чтобы можно было быстро находить все idx по категории
        self.samples_by_category = self.build_samples_by_category()

        # -- Для совместимости с логикой DatasetCOCO (где есть build_class_ids, build_img_metadata и пр.)
        self.class_ids = list(self.samples_by_category.keys())     # все category_id
        self.img_metadata_classwise = self.build_img_metadata_classwise()
        self.img_metadata = self.build_img_metadata()

        print(f"[RefCOCOG] Split = {self.split}, total samples = {len(self.dataset)}")
        print(f"[RefCOCOG] Found {len(self.class_ids)} distinct category_ids.")

    def __len__(self):
        """
        Как в DatasetCOCO:
          - Для обучения (split='trn') берём всю длину
          - Для валидации/теста (split='val') ограничимся 1000, либо возьмём фактическую длину
        """
        if self.split == 'train':
            return len(self.dataset)
        else:
            return min(1000, len(self.dataset))

    def __getitem__(self, idx):
        """
        Возвращаем словарь в том же формате, что и DatasetCOCO __getitem__:
          batch = {
             'query_img': ...,
             'query_mask': ...,
             'query_name': ...,
             'org_query_imsize': ...,
             'support_imgs': ...,
             'support_masks': ...,
             'support_names': ...,
             'class_id': ...,
             'query_text': ...
          }
        где:
          - query_img, support_imgs: тензоры (C,H,W)
          - query_mask, support_masks: тензоры (H,W) (после F.interpolate, если нужно)
          - query_name, support_names: строки (идентификаторы)
          - class_id: категория (category_id)
          - query_text: описание из датасета
        """
        # 1) Берём элемент датасета по idx, это будет наша query
        query_item = self.dataset[idx]
        
        # 2) Получаем информацию о query
        query_img_pil, query_mask_pil, query_name, org_qry_imsize, query_text, cat_id = self._load_image_and_mask(query_item)

        # 3) Ищем другие индексы в той же категории
        if cat_id is None or cat_id not in self.samples_by_category:
            # Если category_id нет (или не в словаре), support будет пустым
            support_imgs_pil = []
            support_masks_pil = []
            support_names = []
        else:
            idx_list = self.samples_by_category[cat_id]
            # Убираем из списка сам idx (чтобы query не дублировалось в support)
            idx_list_for_support = [x for x in idx_list if x != idx]
            # Сэмплируем нужное количество support-примеров
            if len(idx_list_for_support) >= self.shot:
                support_indices = random.sample(idx_list_for_support, self.shot)
            else:
                # Если нет достаточного кол-ва примеров, берём все, что есть
                support_indices = idx_list_for_support

            # 4) Считываем support
            support_imgs_pil = []
            support_masks_pil = []
            support_names = []
            for s_idx in support_indices:
                s_item = self.dataset[s_idx]
                s_img_pil, s_mask_pil, s_name, _, _, _ = self._load_image_and_mask(s_item)
                support_imgs_pil.append(s_img_pil)
                support_masks_pil.append(s_mask_pil)
                support_names.append(s_name)

        # 5) Применяем transform (и F.interpolate) к query
        #    query_img (C,H,W), query_mask (H,W)
        query_img = self._apply_transform_to_query(query_img_pil, query_mask_pil)

        # 6) Применяем transform к support
        support_imgs, support_masks = self._apply_transform_to_support(support_imgs_pil, support_masks_pil)

        # 7) Формируем выходной словарь
        batch = {
            'query_img': query_img['img'],              # torch.Size([3,H,W])
            'query_mask': query_img['mask'],            # torch.Size([H,W])
            'query_name': query_name,                   # строка
            'org_query_imsize': org_qry_imsize,         # (W,H)
            'support_imgs': support_imgs,               # torch.Size([shot,3,H,W])
            'support_masks': support_masks,             # torch.Size([shot,H,W])
            'support_names': support_names,             # список строк
            'class_id': torch.tensor(cat_id if cat_id is not None else -1),
            'query_text': query_text                    # описание
        }
        return batch

    # -------------------------------------------------------------------------
    # Вспомогательные методы
    # -------------------------------------------------------------------------
    def build_samples_by_category(self):
        """
        Сгруппируем индексы датасета по их category_id.
        Ожидается, что в item["category_id"] лежит integer, как в COCO.
        """
        cat_dict = {}
        for i, item in enumerate(self.dataset):
            cat_id = item.get("category_id", None)  # int или None
            if cat_id is None:
                continue
            if cat_id not in cat_dict:
                cat_dict[cat_id] = []
            cat_dict[cat_id].append(i)
        return cat_dict

    def build_class_ids(self):
        """
        Для совместимости, возвращаем список всех category_id
        """
        return list(self.samples_by_category.keys())

    def build_img_metadata_classwise(self):
        """
        Аналог dict: {cat_id: [список индексов]} — уже есть self.samples_by_category
        """
        return self.samples_by_category

    def build_img_metadata(self):
        """
        В DatasetCOCO было список имён файлов, здесь вернём список индексов [0..len(dataset)-1].
        """
        return list(range(len(self.dataset)))

    def _load_image_and_mask(self, item):
        """
        Считываем PIL-изображение, строим PIL-маску из полигона item["raw_anns"]["segmentation"],
        берём текстовое описание (query_text) из item["sentences"] или "raw_anns".
        Возвращаем:
          (pil_image, pil_mask, name_str, (W,H), query_text, category_id)
        """
        # 1) Путь к изображению
        image_path = item["image_path"]  # полный путь или относительный
        name_str = os.path.basename(image_path)

        # 2) Открываем PIL
        pil_image = Image.open(image_path).convert("RGB")
        w, h = pil_image.size

        # 3) Считываем аннотации
        raw_anns = json.loads(item["raw_anns"])
        seg_data = raw_anns.get("segmentation", [])

        # 4) Генерируем PIL-маску
        mask_pil = Image.new('L', (w, h), 0)
        draw = ImageDraw.Draw(mask_pil)
        if isinstance(seg_data, list) and len(seg_data) > 0:
            # Может быть несколько полигонов (список списков) или один
            if isinstance(seg_data[0], list):
                # несколько полигонов
                for seg in seg_data:
                    pts = list(zip(seg[0::2], seg[1::2]))
                    draw.polygon(pts, outline=1, fill=1)
            else:
                # один полигон
                pts = list(zip(seg_data[0::2], seg_data[1::2]))
                draw.polygon(pts, outline=1, fill=1)

        # 5) Берём текст (например, из item["sentences"] или raw_anns["sentences"])
        #    В датасете "jxu124/refcocog" обычно поле "sentences" — список словарей
        #    Смотрим, что у вас реально есть: item["sentences"] или raw_anns["sentences"]
        #    Ниже предполагаем, что query_text = item["sentences"][0]["raw"] если оно есть
        if "sentences" in item and len(item["sentences"]) > 0:
            query_text = item["sentences"][0]["raw"]
        else:
            # fallback
            query_text = ""

        # 6) category_id
        cat_id = item.get("category_id", None)  # int или None

        return pil_image, mask_pil, name_str, (w, h), query_text, cat_id

    def _apply_transform_to_query(self, pil_img, pil_mask):
        """
        Применяем self.transform (Resize/ToTensor/Normalize) к query изображению,
        а маску просто ресайзим через F.interpolate (без нормализации).
        """
        # 1) Если нужно сохранить исходный размер, вы можете разбить transform или не применять Resize.
        #    Ниже — пример логики как в DatasetCOCO:
        #    - Если use_original_imgsize=False, то применяем transform (Resize,...)
        #    - Маску подгоняем к результату query_img.size()
        if not self.use_original_imgsize:
            query_img = self.transform(pil_img)  # (C,H,W)
        else:
            # Даже если хотим сохранить размер, возможно transform включает ToTensor/Normalize — нужно аккуратно:
            query_img = self.transform(pil_img)

        # 2) Маску -> torch.uint8, F.interpolate под тот же (H,W), если не original
        mask_t = torch.tensor(np.array(pil_mask), dtype=torch.uint8)
        if not self.use_original_imgsize:
            _, h_new, w_new = query_img.shape
            mask_t = mask_t.unsqueeze(0).unsqueeze(0).float()  # (1,1,H,W)
            mask_t = F.interpolate(mask_t, size=(h_new, w_new), mode='nearest')
            mask_t = mask_t.squeeze()  # (H_new, W_new)
        else:
            mask_t = mask_t.float()

        return {'img': query_img, 'mask': mask_t}

    def _apply_transform_to_support(self, pil_imgs, pil_masks):
        """
        Аналогичная логика для support. Возвращаем два тензора:
           support_imgs (shot,3,H,W)
           support_masks (shot,H,W)
        """
        support_imgs_t = []
        support_masks_t = []
        for s_img_pil, s_mask_pil in zip(pil_imgs, pil_masks):
            # 1) s_img
            if not self.use_original_imgsize:
                s_img = self.transform(s_img_pil)  # (3,H,W)
            else:
                s_img = self.transform(s_img_pil)

            # 2) s_mask
            s_mask = torch.tensor(np.array(s_mask_pil), dtype=torch.uint8)
            if not self.use_original_imgsize:
                _, h_new, w_new = s_img.shape
                s_mask = s_mask.unsqueeze(0).unsqueeze(0).float()
                s_mask = F.interpolate(s_mask, size=(h_new, w_new), mode='nearest')
                s_mask = s_mask.squeeze()
            else:
                s_mask = s_mask.float()

            support_imgs_t.append(s_img)
            support_masks_t.append(s_mask)

        if len(support_imgs_t) > 0:
            support_imgs_tensor = torch.stack(support_imgs_t, dim=0)   # (shot,3,H,W)
            support_masks_tensor = torch.stack(support_masks_t, dim=0) # (shot,H,W)
        else:
            # Если shot=0 или не нашли support, возвращаем пустые тензоры
            support_imgs_tensor = torch.empty(0)
            support_masks_tensor = torch.empty(0)

        return support_imgs_tensor, support_masks_tensor
