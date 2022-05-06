import streamlit as st 
from PIL import Image
import cv2
import pandas as pd
import numpy as np
from datetime import datetime

@st.cache
def load_image(img):
    im = Image.open(img)
    return im

@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

def detection_visage(img_path):
    cascade_path = "./cascades/haarcascade_frontalface_default.xml"
    # La couleur du carré qui entoure le visage détecté
    color = (0, 255, 0)
    to_image = Image.open(img_path)
    src = cv2.cvtColor(np.array(to_image), cv2.COLOR_RGB2BGR)
    cascade = cv2.CascadeClassifier(cascade_path)
    rect = cascade.detectMultiScale(src)

    tableau = []
    if len(rect) > 0:
        for i, [x, y, w, h] in enumerate(rect):
            # affiche le cadre de détection
            cv2.rectangle(src, (x, y), (x+w, y+h), color,2)
            # affiche le carré
            cv2.rectangle(src, (x, y-30), (x+w, y), color, -1)
            cv2.putText(src, f"personne : {i+1}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0), 1)

            date_now = datetime.today()
            tableau.append([f"Visage {i+1}", f"{date_now.day}/{date_now.month}/{date_now.year}", f"{date_now.hour}:{date_now.minute}:{date_now.second}"])

    
    tableau = pd.DataFrame(tableau, columns=["Personne", "Date", "Heure"])

    return cv2.cvtColor(src, cv2.COLOR_BGR2RGB), tableau



def main():
    """Face Detection App"""
    st.title("Application de détection de visage")
    st.text("Avec Streamlit and OpenCV")
    
    activities = ["Détection image","Autre"]
    choice = st.sidebar.selectbox("Select Activty",activities)
    
    if choice == 'Détection image':
        st.subheader("Détection de visage")
        
        image_file = st.file_uploader("Upload Image",type=['jpg','png','jpeg'])

        if image_file is not None:
            
            if st.button("Détecter visage"):
                st.text("Image originale")
                img, tableau = detection_visage(image_file)
                st.image(img, use_column_width=True)
                st.dataframe(tableau)

                csv = convert_df(tableau)

                st.download_button("Press to Download", csv, "file.csv", "text/csv", key='download-csv')



            else:
                original_image = load_image(image_file)
                st.text("Image chargée")
                st.image(original_image, use_column_width=True)

        

    elif choice == 'Autre':
        st.subheader("Autre")
		

if __name__ == '__main__':
		main()	