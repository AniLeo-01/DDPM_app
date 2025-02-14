# Implementation of DDPM architecture for Diffusion Based Image2Image Generation

This is a comprehensive application for training and deploying Denoising Diffusion Probabilistic Models (DDPM) using Streamlit. It allows users to train diffusion models on custom datasets and perform inference to generate images or videos based on the trained model.

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Launching the Streamlit App](#launching-the-streamlit-app)
  - [Train Page](#train-page)
  - [Inference Page](#inference-page)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Training Interface:** Train DDPM models with customizable parameters and support for custom image datasets.
- **Inference Interface:** Generate single images or videos based on the trained model with adjustable FPS for videos.
- **Streamlit Integration:** User-friendly web interface for interacting with the diffusion model without the need for command-line operations.
- **Custom Dataset Support:** Upload and train on your own image datasets.
- **Sample Saving:** Automatically save generated samples at specified intervals during training.

## Prerequisites

Before setting up the project, ensure you have the following installed:

- **Python 3.8 or higher**
- **CUDA 10.2 or higher** (if using GPU for training)
- **Git** (for cloning the repository)

## Installation

Follow these steps to set up the project locally:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/your-username/DDPM_app.git
   cd DDPM_app
   ```

2. **Create a Virtual Environment:**

   It's recommended to use a virtual environment to manage dependencies.

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Upgrade pip:**

   ```bash
   pip install --upgrade pip
   ```

4. **Install Dependencies:**

   Install all required Python packages using `requirements.txt`.

   ```bash
   pip install -r requirements.txt
   ```

   **Note:** Ensure you have the appropriate CUDA toolkit installed if you plan to utilize GPU acceleration.

## Configuration

The project configuration is managed through the `src/config.py` file. This file contains all the essential parameters for training and inference. Here's a brief overview of the configurable parameters:

**Parameters:**

- `image_size`: The size of the input images (e.g., 28 for 28x28 images).
- `channels`: Number of image channels (1 for grayscale, 3 for RGB).
- `timesteps`: Number of diffusion timesteps.
- `batch_size`: Number of samples per batch during training.
- `epochs`: Number of training epochs.
- `save_and_sample_every`: Interval for saving generated samples during training.
- `learning_rate`: Learning rate for the optimizer.
- `device`: Computational device (`cuda` if available, else `cpu`).

## Usage

### Launching the Streamlit App

To start the Streamlit application, navigate to the project root directory and run:

```bash
streamlit run src/app.py
```

This command will launch the web interface, typically accessible at `http://localhost:8501`.

### Train Page

The **Train** page allows you to train the DDPM model with customizable parameters.

1. **Navigate to the Train Page:**
   - Select "Train" from the sidebar menu.

2. **Configure Training Parameters:**
   - **Image Size:** Specify the size of the input images.
   - **Channels:** Number of image channels (1 for grayscale, 3 for RGB).
   - **Timesteps:** Number of diffusion steps.
   - **Batch Size:** Number of samples per training batch.
   - **Epochs:** Total number of training epochs.
   - **Learning Rate:** Optimizer's learning rate.
   - **Save and Sample Every:** Frequency (in steps) to save generated samples.

3. **Upload Custom Dataset (Optional):**
   - Use the file uploader to upload images (`.png`, `.jpg`, `.jpeg`) as your custom dataset.
   - Ensure that the dataset contains images only.

4. **Start Training:**
   - Click the "Start Training" button to begin the training process.
   - Training progress and loss metrics will be displayed in real-time.
   - Generated samples will be saved at intervals specified by `save_and_sample_every`.

### Inference Page

The **Inference** page enables you to generate images or videos using the trained DDPM model.

1. **Navigate to the Inference Page:**
   - Select "Inference" from the sidebar menu.

2. **Select Inference Type:**
   - **Image:** Generate a single random image.
   - **Video:** Generate a video showcasing the diffusion process.

3. **Generate Image:**
   - Click the "Sample Image" button to generate and display a single image.
   
4. **Generate Video:**
   - Specify the Frames Per Second (FPS) for the video.
   - Click the "Generate Video" button to create and view an MP4 video of the diffusion process.

## Project Structure

Here's an overview of the project's file structure:

```
DDPM_app/
├── README.md
├── requirements.txt
├── src/
    ├── __init__.py
    ├── app.py
    ├── config.py
    ├── train.py
    ├── inference.py
    ├── eval.py
    └── diffusion/
        ├── __init__.py
        ├── model.py
        ├── dataloader.py
        └── utils.py

```

**Descriptions:**

- `README.md`: Project documentation.
- `requirements.txt`: Python package dependencies.
- `src/`: Source code directory.
  - `app.py`: Streamlit application entry point.
  - `config.py`: Configuration parameters.
  - `train.py`: Training script encapsulated in the `train_model` function.
  - `inference.py`: Inference functions for generating images and videos.
  - `eval.py`: Evaluation scripts (implementation details not provided).
  - `diffusion/`: Module containing diffusion model components.
    - `model.py`: Defines the UNet model architecture.
    - `dataloader.py`: Data loading utilities for training and custom datasets.
    - `utils.py`: Utility functions for loss computation, sampling, etc.

### Additional Resources

- **Streamlit Documentation:** [https://docs.streamlit.io/](https://docs.streamlit.io/)
- **PyTorch Documentation:** [https://pytorch.org/docs/stable/index.html](https://pytorch.org/docs/stable/index.html)

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**

2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add Your Feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## Acknowledgements

- [OpenAI](https://www.openai.com/) for their contributions to the AI community.
- [Streamlit](https://streamlit.io/) for providing an excellent framework for building web applications.
- [PyTorch](https://pytorch.org/) for their powerful deep learning library.
- [Denoising Diffusion Papers and Resources](https://github.com/hojonathanho/diffusion) for inspiring the implementation of DDPM.
