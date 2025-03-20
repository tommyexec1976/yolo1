import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

model_path = 'best.pt'            
model = YOLO(model_path)

st.title("ตรวจจับวัตถุ : ภาพนิ่ง")
img_file = st.file_uploader("เปิดไฟล์ภาพ")

if img_file is not None:    
    file_bytes = np.asarray(bytearray(img_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    #----------------------------------------------
    results = model.predict(source=img, device='cpu', save=False, conf=0.3)
    
    for result in results:
        output = result.plot()
   
    #----------------------------------------------
    st.header('ผลการตรวจจับวัตถุ')
    st.image(output, caption='ภาพ Output',channels="BGR")
    
