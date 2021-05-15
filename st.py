import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
import pandas as pd


st.markdown('# Diagnosing Lung X-Rays 🩺')
st.markdown("This model uses a convolutional neural network to classify images of lung x-rays.")
st.markdown("It has been **trained on a Tesla V100-SXM2 GPU using ~4500 different lung x-rays**.")
st.markdown("The model achieved an accuracy of ~91% on a testset of 600 lung x-rays")
st.markdown("*Disclaimer: For educational porpuses only*")
uploaded_file = st.file_uploader('Upload image...', type=['jpeg', 'jpg', 'png'])

if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, caption='This x-ray will be diagnosed...', use_column_width=True)

	