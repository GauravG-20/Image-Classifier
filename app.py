import pandas as pd
import streamlit as st
from PIL import Image
import numpy as np
import torchvision.transforms as T
import torch
import time

@st.cache_resource(ttl=3600)
def load_model():
    learn_inf = torch.jit.load("checkpoints/original_exported.pt")
    return learn_inf

def classify_image(model, img):
    # Transform the image to tensor
    timg = T.ToTensor()(img).unsqueeze_(0)

    # Calling the model
    softmax = model(timg).data.cpu().numpy().squeeze()

    # Get the indexes of the classes ordered by softmax (larger first)
    idxs = np.argsort(softmax)[::-1]

    # Return top 5 classes and probabilities
    top_classes = [(model.class_names[idx], softmax[idx]) for idx in idxs[:5]]
    return top_classes

def main():
    st.title("Image Classifier")

    # Load the model
    with st.spinner('Loading the model...'):
        model = load_model()

    with st.spinner('Loading the image...'):
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

        if uploaded_image is not None:
            img = Image.open(uploaded_image)
            st.image(img, caption="Uploaded Image", use_column_width=True)

            if st.button("Classify"):
                with st.spinner("Classifying..."):

                    # Classify the image
                    top_classes = classify_image(model, img)

                    # Display the top classes and probabilities
                    data,value = [],[]
                    tab1,tab2 = st.tabs(["Classification","Bar Chart"])
                    with tab1:
                        for i, (class_name, probability) in enumerate(top_classes, start=1):
                            name = class_name.split(".")[-1]
                            name = name.replace("_", " ")
                            st.write(f"{i}. {name} (Probability: {probability*100:.2f}%)")
                            data.append(name)
                            value.append(probability)
                        
                    chart_data = pd.DataFrame({
                        'data':data,
                        'values':value
                    })

                    chart_data.sort_values(by='values',inplace=True)
                    chart_data.reset_index(inplace=True)
                    
                    with tab2:
                        st.bar_chart(chart_data,x='data',y='values')

if __name__ == "__main__":
    main()
