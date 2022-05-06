import streamlit as st 
from PIL import Image
from image_detect import detection_visage
import cv2


@st.cache
def load_image(img):
    im = Image.open(img)
    return im

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

            else:
                original_image = load_image(image_file)
                st.text("Image chargée")
                st.image(original_image, use_column_width=True)

        

    elif choice == 'Autre':
        st.subheader("Autre")
		

if __name__ == '__main__':
		main()	
