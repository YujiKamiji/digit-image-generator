import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms.functional as F

# --- Definir a mesma arquitetura do Gerador do Colab ---
latent_dim = 100
n_classes = 10
img_shape = (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.Linear(latent_dim + n_classes, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1 * 28 * 28),
            nn.Tanh()
        )
    def forward(self, z, labels):
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(-1, *img_shape)

# --- Carregar o Modelo Treinado ---
# Coloque o modelo para rodar na CPU
device = torch.device("cpu")
generator = Generator().to(device)
# Carregar os pesos salvos (o arquivo generator.pth deve estar no mesmo diretório)
generator.load_state_dict(torch.load('generator.pth', map_location=device))
generator.eval() # Colocar o modelo em modo de avaliação

# --- Interface do Streamlit ---
st.title("Gerador de Imagens de Dígitos")
st.write("Escolha um dígito (0-9) e o modelo irá gerar 5 imagens manuscritas.")

# Dropdown para escolher o dígito
selected_digit = st.selectbox("Escolha um dígito para gerar:", list(range(10)))

# Botão para gerar as imagens
if st.button("Gerar Imagens"):
    with st.spinner("Gerando imagens..."):
        # Gerar 5 imagens
        n_images = 5
        z = torch.randn(n_images, latent_dim, device=device)
        labels = torch.LongTensor([selected_digit] * n_images).to(device)
        
        generated_imgs = generator(z, labels)
        
        # Exibir as imagens
        cols = st.columns(n_images)
        for i, img_tensor in enumerate(generated_imgs):
            # Desnormalizar a imagem (de [-1, 1] para [0, 1])
            img_tensor = (img_tensor + 1) / 2
            # Converter para um formato que o st.image entende
            img_pil = F.to_pil_image(img_tensor)
            with cols[i]:
                st.image(img_pil, caption=f'Amostra {i+1}', use_column_width=True)