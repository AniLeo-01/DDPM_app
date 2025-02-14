import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from src.train import train_model
from src.inference import sample_image, create_sample_grid_video
from src.config import (
    image_size,
    channels,
    timesteps,
    batch_size,
    epochs,
    learning_rate,
    save_and_sample_every,
    device
)
from src.diffusion.dataloader import get_custom_dataloader
import torch

def main():
    st.title("Diffusion Model Interface")
    
    pages = ["Train", "Inference"]
    page = st.sidebar.selectbox("Select Page", pages)
    
    if page == "Train":
        st.header("Train Diffusion Model")
        
        # Training parameters
        with st.form("train_form"):
            input_image_size = st.number_input("Image Size", value=image_size, min_value=1)
            input_channels = st.number_input("Channels", value=channels, min_value=1)
            input_timesteps = st.number_input("Timesteps", value=timesteps, min_value=1)
            input_batch_size = st.number_input("Batch Size", value=batch_size, min_value=1)
            input_epochs = st.number_input("Epochs", value=epochs, min_value=1)
            input_learning_rate = st.number_input("Learning Rate", value=learning_rate, format="%.5f")
            input_save_and_sample_every = st.number_input("Save and Sample Every", value=save_and_sample_every, min_value=1)
            upload_dataset = st.file_uploader("Upload Custom Dataset (Images Only)", accept_multiple_files=True, type=["png", "jpg", "jpeg"])
            submit = st.form_submit_button("Start Training")
        
        if submit:
            if upload_dataset:
                dataloader = get_custom_dataloader(upload_dataset, input_batch_size, device)
                st.success("Custom dataset loaded.")
            else:
                dataloader = None  # Use default dataloader
            train_model(
                image_size=input_image_size,
                channels=input_channels,
                timesteps=input_timesteps,
                batch_size=input_batch_size,
                epochs=input_epochs,
                learning_rate=input_learning_rate,
                save_and_sample_every=input_save_and_sample_every,
                dataloader_override=dataloader
            )
            st.success("Training completed.")
    
    elif page == "Inference":
        st.header("Inference with Trained Model")
        
        inference_option = st.selectbox("Select Inference Type", ["Image", "Video"])
        if inference_option == "Image":
            if st.button("Sample Image"):
                sample_image()
                st.image("sample_image.png", caption="Generated Image")
        elif inference_option == "Video":
            fps = st.number_input("Frames Per Second", value=24, min_value=1)
            if st.button("Generate Video"):
                create_sample_grid_video(timesteps, fps=fps)
                st.video("diffusion.mp4")

if __name__ == "__main__":
    main() 