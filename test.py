import onnxruntime as ort
import numpy as np
from PIL import Image

# ---- 1. 加载模型 ----
onnx_model_path = "outputs/onnx_models/brain_tumor_segmentation.onnx"
session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
input_shape = session.get_inputs()[0].shape  # [1,1,H,W]
print("模型输入:", input_shape)

# ---- 2. 读取并预处理图像 ----
img_path = "data/2D copy/test/images/train_0001 copy 2.png"
img = Image.open(img_path).convert("L")

H, W = input_shape[2], input_shape[3]
img_resized = img.resize((W, H))

# 不做归一化，直接用原始像素值
img_np = np.array(img_resized).astype(np.float32)   # [H,W]
img_input = np.expand_dims(img_np, axis=(0,1))      # [1,1,H,W]

# ---- 3. 推理 ----
outputs = session.run([output_name], {input_name: img_input})
logits = outputs[0]  # [1,num_classes,H,W]

# ---- 4. 后处理 ----
probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
pred_mask = np.argmax(probs, axis=1)[0].astype(np.uint8)  # [H,W]

# ×80 映射不同类别到灰度值
scaled_mask = (pred_mask * 80).astype(np.uint8)

# ---- 5. 保存结果 ----
mask_img = Image.fromarray(scaled_mask)
mask_img.save("pred_mask.png")

