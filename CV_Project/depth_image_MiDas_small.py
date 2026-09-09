import torch
import cv2
import matplotlib.pyplot as plt


# -----------------------------
# 1. Загружаем модель MiDaS
# -----------------------------

model_type = "MiDaS_small"

midas = torch.hub.load(
    "intel-isl/MiDaS",
    model_type
)


# -----------------------------
# 2. Выбираем устройство
# -----------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

midas.to(device)

midas.eval()


# -----------------------------
# 3. Загружаем transforms
# -----------------------------

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)

transform = midas_transforms.small_transform


# -----------------------------
# 4. Загружаем изображение
# -----------------------------

image = cv2.imread(
    "depth_test.jpg"
)

if image is None:
    raise FileNotFoundError(
        "Не найден depth_test.jpg"
    )


# OpenCV читает BGR,
# а модели обычно нужен RGB

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# -----------------------------
# 5. Подготавливаем изображение
# -----------------------------

input_batch = transform(
    image_rgb
).to(device)


# -----------------------------
# 6. Inference
# -----------------------------

with torch.no_grad():

    prediction = midas(
        input_batch
    )


    prediction = torch.nn.functional.interpolate(

        prediction.unsqueeze(1),

        size=image_rgb.shape[:2],

        mode="bicubic",

        align_corners=False

    ).squeeze()


# -----------------------------
# 7. Tensor -> NumPy
# -----------------------------

depth_map = prediction.cpu().numpy()


# -----------------------------
# 8. Визуализация
# -----------------------------

plt.figure(figsize=(12, 6))


plt.subplot(1, 2, 1)

plt.imshow(image_rgb)

plt.title("Original Image")

plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(
    depth_map,
    cmap="inferno"
)

plt.title("MiDaS Depth Map")

plt.axis("off")


plt.show()