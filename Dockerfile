# 1. Image de base imposée par AMD (Supporte GPU CDNA / RDNA & PyTorch ROCm)
FROM rocm/pytorch:rocm10.0_ubuntu26.04_py3.14_pytorch_release_2.13.0

# Interdiction de fusionner les couches (pas de docker build --squash ou FROM scratch)
# pour préserver l'identité des couches ROCm vérifiée automatiquement par l'évaluateur.

# 2. Définition du répertoire de travail requis
WORKDIR /app

# 3. Copie et installation des dépendances Python
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# 4. Création de la structure des dossiers exigée par le pipeline d'évaluation
RUN mkdir -p /app/input /app/output /models

# 5. Copie du code applicatif
COPY app/app.py /app/app.py

# 6. Pré-téléchargement des poids du modèle VLM dans l'image (/models ou cache HF)
# Cela évite de dépendre du réseau au démarrage et garantit un lancement immédiat (< 10 min)
RUN python3 -c "\
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration; \
model_id = 'Qwen/Qwen2-VL-2B-Instruct'; \
AutoProcessor.from_pretrained(model_id); \
Qwen2VLForConditionalGeneration.from_pretrained(model_id); \
"

# 7. Variables d'environnement pour l'optimisation ROCm / PyTorch
ENV PYTHONUNBUFFERED=1