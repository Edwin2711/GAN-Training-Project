import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

# ====================================================================
# 1. ARQUITECTURA DEL MODELO (debe ser IDÉNTICA a la del entrenamiento)
# ====================================================================
# Parámetros (deben coincidir con el entrenamiento)
noise_dim = 100
num_classes = 10
embedding_dim = 10
image_size = 28 * 28

# Definimos la clase del Generador de nuevo
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_embedding = nn.Embedding(num_classes, embedding_dim)
        self.model = nn.Sequential(
            nn.Linear(noise_dim + embedding_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, image_size),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        c = self.label_embedding(labels)
        x = torch.cat([noise, c], 1)
        out = self.model(x)
        return out.view(-1, 1, 28, 28)

# ====================================================================
# 2. CARGAR EL MODELO ENTRENADO
# ====================================================================
# Usar 'cpu' ya que el entorno de despliegue de Streamlit no tiene GPU
device = torch.device('cpu')
model = Generator().to(device)

# Cargar los pesos guardados.
# Asegúrate de que el archivo 'generator.pth' esté en el mismo directorio.
try:
    model.load_state_dict(torch.load('generator.pth', map_location=device))
    model.eval() # Poner el modelo en modo de evaluación
except FileNotFoundError:
    st.error("Error: El archivo del modelo 'generator.pth' no se encontró.")
    st.stop()


# ====================================================================
# 3. INTERFAZ DE USUARIO DE STREAMLIT
# ====================================================================
st.set_page_config(layout="wide")

st.title("Handwritten Digit Image Generator")
st.write(
    "Generate synthetic MNIST-like images using your trained model. "
    "This app uses a Conditional GAN (cGAN) trained on the MNIST dataset."
)

# --- Controles del usuario ---
st.sidebar.header("Controls")
selected_digit = st.sidebar.selectbox("Choose a digit to generate (0-9):", list(range(10)))

if st.sidebar.button("Generate Images"):
    st.subheader(f"Generated images of digit {selected_digit}")

    # Usar columnas para mostrar las 5 imágenes una al lado de la otra
    cols = st.columns(5)
    
    for i in range(5):
        with torch.no_grad():
            # Generar ruido y etiqueta
            noise = torch.randn(1, noise_dim, device=device)
            label = torch.LongTensor([selected_digit]).to(device)
            
            # Generar la imagen
            generated_image_tensor = model(noise, label)
            
            # Post-procesar la imagen para mostrarla
            # 1. Quitar dimensiones de batch y canal: .squeeze()
            # 2. Mover a CPU y convertir a numpy: .cpu().numpy()
            # 3. Desnormalizar de [-1, 1] a [0, 255]
            img_np = generated_image_tensor.squeeze().cpu().numpy()
            img_np = (img_np + 1) / 2 * 255
            img_np = img_np.astype(np.uint8)
            
            # Mostrar la imagen en su columna
            with cols[i]:
                st.image(img_np, caption=f"Sample {i + 1}", use_column_width=True)
else:
    st.info("Select a digit and click 'Generate Images' in the sidebar.")
