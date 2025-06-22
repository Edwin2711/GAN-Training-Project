import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from urllib.request import urlopen

# --- MODEL DEFINITION ---
# IMPORTANT: The model class here MUST be identical to the one used for training.
LATENT_DIM = 100
N_CLASSES = 10
IMG_SIZE = 28
CHANNELS = 1
IMG_SHAPE = (CHANNELS, IMG_SIZE, IMG_SIZE)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(N_CLASSES, LATENT_DIM)
        self.model = nn.Sequential(
            nn.Linear(LATENT_DIM * 2, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, IMG_SIZE * IMG_SIZE * CHANNELS),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_embedding(labels)
        x = torch.cat([noise, c], 1)
        img = self.model(x)
        img = img.view(img.size(0), *IMG_SHAPE)
        return img

# --- MODEL LOADING ---
# Use Streamlit's caching to load the model only once.
@st.cache_resource
def load_model():
    # IMPORTANT: Replace this with the raw URL of your generator.pth file on GitHub
    MODEL_URL = "https://raw.githubusercontent.com/Edwin2711/GAN-Training-Project/main/generator.pth"
    
    # Use CPU for inference as most free deployment platforms don't provide GPUs
    device = torch.device('cpu')
    
    model = Generator()
    
    # Download the state dictionary
    try:
        model_file = urlopen(MODEL_URL)
        state_dict = torch.load(model_file, map_location=device)
        model.load_state_dict(state_dict)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    model.to(device)
    model.eval() # Set the model to evaluation mode
    return model

generator = load_model()

# --- WEB APP INTERFACE ---
st.set_page_config(layout="wide")

st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")
st.write("---")

# User input
st.sidebar.header("Controls")
selected_digit = st.sidebar.selectbox("Choose a digit to generate (0-9):", list(range(10)))
generate_button = st.sidebar.button("Generate Images", type="primary")

if generate_button and generator is not None:
    st.subheader(f"Generated images of digit {selected_digit}")

    # Create 5 columns for the images
    cols = st.columns(5)

    for i in range(5):
        # Generate a new image
        with torch.no_grad():
            # Create random noise and the desired label
            noise = torch.randn(1, LATENT_DIM, device='cpu')
            label = torch.LongTensor([selected_digit]).to('cpu')
            
            # Generate the image
            generated_image = generator(noise, label)
            
            # Post-process for display: scale from [-1, 1] to [0, 1]
            generated_image = 0.5 * generated_image + 0.5
            # Convert to numpy array
            np_image = generated_image.squeeze().numpy()

        # Display in the respective column
        with cols[i]:
            st.image(np_image, caption=f"Sample {i+1}", use_column_width='always')

elif generator is None:
    st.error("Model could not be loaded. Please check the model URL and file.")
else:
    st.info("Select a digit and click 'Generate Images' to start.")
