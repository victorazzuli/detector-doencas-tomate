import json

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

TAM = 224

st.set_page_config(page_title="Detector de Doenças do Tomate", page_icon="🍅", layout="centered")


@st.cache_resource
def carregar_modelo():
    modelo = tf.keras.models.load_model("modelo_tomate.keras")
    with open("classes.json", encoding="utf-8") as f:
        classes = json.load(f)
    return modelo, classes


modelo, classes = carregar_modelo()

st.title("🍅 Detector de Doenças do Tomate")
st.write(
    "Envie a foto de uma folha de tomate e a IA aponta se ela está saudável "
    "ou qual doença é mais provável."
)

arquivo = st.file_uploader("Escolha uma foto da folha", type=["jpg", "jpeg", "png"])

if arquivo is not None:
    imagem = Image.open(arquivo).convert("RGB")
    st.image(imagem, caption="Foto enviada", width="stretch")

    with st.spinner("Analisando a folha..."):
        entrada = np.expand_dims(np.array(imagem.resize((TAM, TAM)), dtype=np.float32), 0)
        probs = modelo.predict(entrada, verbose=0)[0]

    ordem = probs.argsort()[::-1]
    melhor = ordem[0]
    nome = classes[melhor]

    st.subheader("Resultado")
    if "healthy" in nome.lower():
        st.success(f"**Folha saudável** ({probs[melhor]:.0%} de confiança)")
    else:
        st.error(f"**{nome}** ({probs[melhor]:.0%} de confiança)")

    if probs[melhor] < 0.6:
        st.warning(
            "A confiança está baixa. Tente uma foto mais nítida, só com a folha, "
            "de preferência com fundo simples."
        )

    st.write("**Outras possibilidades:**")
    for i in ordem[1:4]:
        st.progress(float(probs[i]), text=f"{classes[i]}: {probs[i]:.1%}")

st.divider()
st.caption(
    "Projeto de portfólio · Modelo MobileNetV2 treinado com o dataset PlantVillage. "
    "Ferramenta educacional, não substitui a avaliação de um agrônomo."
)
