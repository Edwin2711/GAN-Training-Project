import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import os # Importamos 'os' para verificar si el archivo existe

# --- MODEL DEFINITION ---
# La definición de la clase del modelo debe ser idéntica a la usada en el entrenamiento.
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

# --- MODEL LOADING (VERSIÓN CORREGIDA) ---
# Esta función ahora carga el modelo desde un archivo local en el repositorio.
@st.cache_resource
def load_model():
    MODEL_PATH = "generator.pth"
    device = torch.device('cpu')
    model = Generator()

    # Verificar si el archivo del modelo existe antes de intentar cargarlo
    if not os.path.exists(MODEL_PATH):
        st.error(f"Error: El archivo del modelo '{MODEL_PATH}' no se encontró. "
                 "Asegúrate de que esté en el repositorio de GitHub junto a app.py.")
        return None

    try:
        # Cargar el modelo directamente desde la ruta del archivo local
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() # Poner el modelo en modo de evaluación
        return model
    except Exception as e:
        st.error(f"Ocurrió un error al cargar el modelo: {e}")
        return None

generator = load_model()

# --- WEB APP INTERFACE ---
st.set_page_config(layout="wide")

st.title("Handwritten Digit Image Generator")
st.write("Generate synthetic MNIST-like images using your trained model.")
st.write("---")

# Controles en la barra lateral
st.sidebar.header("Controls")
selected_digit = st.sidebar.selectbox("Choose a digit to generate (0-9):", list(range(10)))
generate_button = st.sidebar.button("Generate Images", type="primary")


if generator is not None:
    if generate_button:
        st.subheader(f"Generated images of digit {selected_digit}")

        # Crear 5 columnas para las imágenes
        cols = st.columns(5)

        for i in range(5):
            # Generar una nueva imagen
            with torch.no_grad():
                noise = torch.randn(1, LATENT_DIM, device='cpu')
                label = torch.LongTensor([selected_digit]).to('cpu')
                generated_image = generator(noise, label)
                
                # Post-procesar para mostrar: escalar de [-1, 1] a [0, 1]
                generated_image = 0.5 * generated_image + 0.5
                np_image = generated_image.squeeze().numpy()

            # Mostrar en la columna respectiva
            with cols[i]:
                st.image(np_image, caption=f"Sample {i+1}", use_column_width='always')
    else:
        st.info("Select a digit and click 'Generate Images' to start.")
else:
    # Este mensaje se mostrará si la carga del modelo falla
    st.error("The application could not start because the model failed to load.")