from pathlib import Path
import numpy as np
import streamlit as st
import torch
from PIL import Image

st.set_page_config(layout="wide")

root_path = Path('./models')
device = 'cpu'

models_paths = {
    'unet': 'qartezator_unet.pt',
    'pix2pix': 'qartezator_pix2pix.pt',
    'cyclegan': 'qartezator_cyclegan.pt',
    'discogan': 'qartezator_discogan.pt',
    'lama': 'qartezator_lama.pt'
}

@st.cache_resource
def load_models(models_paths):
    models = {}
    for model_name, model_path in models_paths.items():
        model = torch.jit.load(root_path / model_path, map_location=device)
        model.eval()
        models[model_name] = model
    return models

models = load_models(models_paths)

def process(model: torch.nn.Module, image: Image.Image, pad_out_to_modulo=32) -> Image.Image:
    """Process given image.
​
    Args:
        model (torch.nn.Module): Loaded model.
        image (PIL.Image.Image): Input image.
        pad_out_to_modulo (int): What integer multiple should image width and height be.
​
    Returns:
        PIL.Image.Image:
    """
    w, h = image.size
    max_side = 1024
    if max(h, w) > max_side:
        resize_ratio = max_side / max(h, w)
        w, h = int(resize_ratio * w), int(resize_ratio * h)
        image = image.resize((w, h), Image.BICUBIC)
    image = image.convert('RGB')
    padding_mode = 'reflect'
    padded_h = (h // pad_out_to_modulo + 1) * pad_out_to_modulo if h % pad_out_to_modulo != 0 else h
    padded_w = (w // pad_out_to_modulo + 1) * pad_out_to_modulo if w % pad_out_to_modulo != 0 else w
    image = np.pad(image, ((0, padded_h - h), (0, padded_w - w), (0, 0)), mode=padding_mode)
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
    image = image / 255.0
    image = image.to(device)
    with torch.no_grad():
        result = model(image)
    result = result.squeeze().permute(1, 2, 0).clip(0, 1).detach().cpu().numpy()
    result = np.uint8(255.0 * result)
    result = result[:h, :w]
    result = Image.fromarray(result)
    return result

st.title("🌍 Qartezator - Aerial Image to Map Translator")

option = st.selectbox(
    'Choose model',
    ('Unet', 'Pix2pix', 'CycleGAN', 'DiscoGAN', 'LaMa'))

# Mock function 
def predict(image):
    if option == 'Unet':
        return process(models['unet'], image, pad_out_to_modulo=32)
    if option == 'Pix2pix':
        return process(models['pix2pix'], image, pad_out_to_modulo=32)
    if option == 'CycleGAN':
        return process(models['cyclegan'], image, pad_out_to_modulo=32)
    if option == 'DiscoGAN':
        return process(models['discogan'], image, pad_out_to_modulo=32)
    if option == 'LaMa':
        return process(models['lama'], image, pad_out_to_modulo=8)

st.write("Upload an aerial image and see the predicted map")

uploaded_file = st.file_uploader("Choose an aerial image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1.image(image, caption="Uploaded Image", use_column_width=True)
    predicted_map = predict(image)
    col2.image(predicted_map, caption="Predicted Map", use_column_width=True)
