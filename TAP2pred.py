import os
import json
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset
from tokenize_anything import model_registry
from tokenize_anything.utils.image import im_rescale, im_vstack
import torch.nn as nn

def show_mask(mask, ax, color=None):
    """
    Наложение маски на изображение.
    mask: numpy массив c формой [H, W] (bool или 0/1).
    ax: matplotlib axes.
    color: np.array [4], например [R, G, B, alpha] (значения от 0 до 1).
    """
    if color is None:
        # Если цвет не задан, генерируем случайный
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    ax.imshow(mask.reshape(mask.shape[-2:] + (1,)) * color.reshape(1, 1, -1))


class TAP_pred(nn.Module):
    """
    Класс, инкапсулирующий логику инференса модели Tokenize Anything (TAP).
    Выполняет:
      1. Загрузку модели и весов концептов в конструкторе.
      2. При вызове `predict` принимает готовое изображение и аннотации (segmentation).
      3. Предобрабатывает изображение, строит маски, текстовые подписи и концепты.
      4. Имеет метод визуализации результатов.
    """

    def __init__(
            self,
            model_type="tap_vit_b",
            checkpoint="./tokenize_anything/weights/tap_vit_b_v1_1.pkl",
            concept_weights="./tokenize_anything/concepts/merged_2560.pkl",
            output_dir="./outputs"
    ):
        """
        Параметры:
            model_type: тип модели (например, "tap_vit_l")
            checkpoint: путь к файлу с весами модели
            concept_weights: путь к pickle-файлу с весами концептов
            output_dir: путь для сохранения результатов (картинка с наложенными масками и текстом)
        """
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_type = model_type
        self.checkpoint = checkpoint
        self.concept_weights = concept_weights
        self.output_dir = output_dir

        # Убедимся, что выходная директория существует
        os.makedirs(self.output_dir, exist_ok=True)

        # Инициализация модели
        print("Initializing TAP model...")
        self.model = self._load_model()
        print("Model initialized.")

    def _load_model(self):
        """
        Внутренний метод для загрузки модели и установки весов концептов.
        """
        # 1. Инициализация модели из реестра
        model = model_registry[self.model_type](checkpoint=self.checkpoint).to(self.device)

        # 2. Загрузка весов концептов
        model.concept_projector.reset_weights(self.concept_weights)

        # 3. Сброс кэша декодера текста (опционально настраиваем размер батча)
        model.text_decoder.reset_cache(max_batch_size=8)
        model.to('cuda')
        model.eval()
        return model

    def forward(self, image, protos):
        """
        Запускает инференс для одного изображения с использованием векторов прототипов.

        Параметры:
            image: numpy массив (H, W, 3) или PIL.Image
            protos: тензор Torch с векторами прототипов (N, D), где N — количество прототипов, D — размерность каждого вектора.

        Возвращает:
            results: словарь с ключами:
                {
                'mask': numpy-маска предсказаний (shape [H, W]),
                'concept': предсказанный концепт (str),
                'caption': сгенерированный текст (str),
                'iou': IOU от модели (float),
                'score': скор концепта (float),
                'image': исходное изображение (numpy, shape (H, W, 3))
                }
        """
        # Преобразуем PIL в numpy, если нужно
        image=image.squeeze().permute(1, 2, 0).cpu().numpy()
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        # Убеждаемся, что изображение в формате (H, W, 3)
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Ожидается изображение формата (H, W, 3)., got: {image.shape}")

        # Подготовка изображений (рескейл)
        image = (image * 255).astype(np.uint8)
        img_list, img_scales = im_rescale(image, scales=[1024], max_size=1024)
        input_size, original_size = img_list[0].shape, image.shape[:2]

        # Подготовка батча
        img_batch = im_vstack(
            img_list, 
            fill_value=self.model.pixel_mean_value, 
            size=(1024, 1024)
        )

        inputs = self.model.get_inputs({"img": img_batch})
        inputs.update(self.model.get_features(inputs))

        # Убеждаемся, что прототипы находятся на правильном устройстве
        protos = protos.to(self.device)

        # Добавляем прототипы в словарь inputs
        inputs["protos"] = protos

        outputs = self.model.get_outputs(inputs)
        iou_score = outputs["iou_pred"]
        mask_pred = outputs["mask_pred"]
        iou_score = torch.cat([
            iou_score[:, :1],
            iou_score[:, 1:] - 1000.0
        ], dim=1)
        mask_index = torch.arange(iou_score.shape[0]), iou_score.argmax(1)
        masks = mask_pred[mask_index]  # [1, H, W]
        masks = self.model.upscale_masks(masks[:, None], img_batch.shape[1:-1])
        masks = masks[..., : input_size[0], : input_size[1]].clone()
        confidence_masks = torch.sigmoid(self.model.upscale_masks(masks, original_size))
        masks = self.model.upscale_masks(masks, original_size).gt(0)
        sem_tokens, sem_embeds = outputs["sem_tokens"], outputs["sem_embeds"]
        concepts, scores = self.model.predict_concept(sem_embeds[mask_index])
        captions = self.model.generate_text(sem_tokens[mask_index])

        # Извлекаем scalar-значения
        iou_val = float(iou_score[mask_index])
        score_val = float(scores)
        caption_val = captions.flatten()[0]
        concept_val = concepts.flatten()[0]

        results = {
            "mask": masks,            # предсказанные маски (shape [1, H, W], но может быть [H, W])
            "low_masks":confidence_masks,
            "concept": concept_val,   # строковый концепт
            "caption": caption_val,   # подпись к объекту
            "iou": iou_val,           # iou от модели
            "score": score_val,       # скор концепта
            "image": image            # исходное изображение (numpy)
        }

        return results

    def visualize_result(self, results, out_filename="result_overlay.png"):
        """
        Визуализирует результат инференса (исходная маска + предсказанная маска)
        и сохраняет в self.output_dir/out_filename.
        """
        img = results["image"]
        orig_mask = results["orig_mask"]
        pred_mask = results["mask"]
        concept_val = results["concept"]
        iou_val = results["iou"]
        score_val = results["score"]
        caption_val = results["caption"]
        polygon_coords = results["polygon_coords"]

        # Если модель вернула shape [1, H, W], то берём 0-й элемент
        if len(pred_mask.shape) == 3 and pred_mask.shape[0] == 1:
            pred_mask = pred_mask[0]

        vis_text = f"{concept_val} ({iou_val:.2f}, {score_val:.2f}): {caption_val}"

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img)
        ax.set_axis_off()

        # Отрисовываем исходную маску (красная) и предсказанную моделью (случайный цвет)
        show_mask(orig_mask, ax, color=np.array([1.0, 0.0, 0.0, 0.4]))
        show_mask(pred_mask, ax)

        # Подпись в левом верхнем углу, ориентируясь на минимум x,y полигона
        x_min, y_min = polygon_coords.min(axis=0)
        ax.text(
            x_min, y_min,
            vis_text,
            fontsize=5,
            color='white',
            bbox=dict(facecolor='black', alpha=0.6, pad=2)
        )

        out_path = os.path.join(self.output_dir, out_filename)
        plt.savefig(out_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        print(f"Result saved to: {out_path}")


if __name__ == "__main__":
    """
    Пример использования класса TAP_pred:
      1) Инициализируем модель
      2) Загружаем/формируем сами картинку (image) и аннотацию (segmentation) вне класса
      3) Вызываем predict и visualize_result
    """

    # Пример инициализации
    tap_model = TAP_pred(
        model_type="tap_vit_l",
        checkpoint="./tokenize_anything/weights/tap_vit_l_v1_1.pkl",
        concept_weights="./tokenize_anything/concepts/merged_2560.pkl",
        output_dir="./outputs"
    )

    dataset = load_dataset("jxu124/refcocog", split="train")
    print("Extracting specific row...")
    data = dataset[0]  # Пример: берем элемент с индексом 25
    print(data.keys())
    image_path = data["image_path"]
    # Указываем реальный путь к изображению (при необходимости скорректируйте)
    image_full_path = os.path.join('./tokenize-anything/', os.path.basename(image_path))
    print("Loading image from dataset...")
    pil_image = Image.open(image_full_path).convert("RGB")
    image = np.array(pil_image)  # Переводим PIL -> numpy (H,W,3)
    # Получаем аннотации
    raw_anns = json.loads(data["raw_anns"])
    # raw_anns['segmentation'] содержит координаты полигона (или список полигонов)
    segmentation = raw_anns['segmentation']
    # Предикт
    results = tap_model.predict(image, segmentation)

    # Визуализация
    tap_model.visualize_result(results, out_filename="example_overlay.png")
