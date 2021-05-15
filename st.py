import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from PIL import Image
import numpy as np
import pandas as pd
import Pneumonia_detection_net

#🩺
st.markdown('# ML application for pneumonia detection ')
st.markdown("This model uses a convolutional neural network to classify images of lung x-rays.")
##st.markdown("It has been **trained on a Tesla V100-SXM2 GPU using ~4500 different lung x-rays**.")
#st.markdown("The model achieved an accuracy of ~91% on a testset of 600 lung x-rays") # BUT YOU HAVE TO CALCULATE IT!!!
st.markdown("*Disclaimer: For educational porpuses only*")
uploaded_file = st.file_uploader('Upload image...', type=['jpeg', 'jpg', 'png'])
def predict(image):
	net = Pneumonia_detection_net.nn_for_pneumonia_detection()
	prediction = net.to_predict(image)
	prediction = prediction.detach().numpy()

	df = pd.DataFrame(data=prediction[0], index=['Bacterial', 'Normal', 'Viral'], columns=['confidence'])

	st.write(f'''### 👍 The probability of a normal lung condition:**{np.round(prediction[0][1], 3)}**''')
	st.write(f'''### 🧫 The probability of bacterial pneumonia:  **{np.round(prediction[0][0], 3)}**''')
	st.write(f'''### 🦠 The probability of viral pneumonia: **{np.round(prediction[0][2], 3)}**''')
	
	st.write('')
	st.bar_chart(df)
if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, caption='This x-ray will be diagnosed...', use_column_width=True)

	if st.button('Predict 🧠'):
		predict(image)