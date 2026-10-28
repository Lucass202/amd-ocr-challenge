import argparse
import json
import os
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

# 1. Sélection du modèle VLM (utilisable sur GPU AMD via ROCm / PyTorch)
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"  # Ou "Qwen/Qwen2-VL-7B-Instruct"

def load_model():
    """Charge le modèle et le processeur en mémoire VRAM."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    return model, processor, device

def run_ocr(image_path: Path, model, processor, device: str) -> str:
    """Effectue la détection et la transcription du texte selon le contrat d'évaluation."""
    # Support direct des formats PNG, JPEG et TIFF via PIL
    image = Image.open(image_path).convert("RGB")

    # Prompt explicite guidant le modèle sur les spécificités du défi
    prompt = (
        "Transcris le texte visible sur cette image selon ces règles strictes :\n"
        "1. Pour une plaque américaine : extrais uniquement le numéro principal. Ignore le nom de l'État (ex: CALIFORNIA) et les slogans.\n"
        "2. Pour une plaque chinoise : conserve la province et la lettre (ex: 京 A·12345 ou 沪 B·88888).\n"
        "3. Pour un panneau routier : lis toutes les lignes de haut en bas en les séparant par un espace."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Préparation du prompt d'entrée
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Inférence GPU
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        text_output = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    return text_output

def main():
    # Gestion de l'argument d'entrée exigé par le système de test (--input-image)
    parser = argparse.ArgumentParser(description="Script d'inférence OCR - AMD AI Challenge")
    parser.add_argument("--input-image", type=str, required=True, help="Chemin vers l'image d'entrée")
    args = parser.parse_args()

    input_image_path = Path(args.input_image)
    
    # Validation du fichier d'entrée
    if not input_image_path.exists():
        raise FileNotFoundError(f"Image introuvable : {input_image_path}")

    # Chargement du modèle et exécution
    model, processor, device = load_model()
    raw_text = run_ocr(input_image_path, model, processor, device)

    # Répertoire de sortie imposé (/app/output/)
    # Utilise /app/output si le dossier /app existe (dans Docker), sinon le dossier local "output"
    output_dir = Path("/app/output") if Path("/app").exists() else Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Nom du fichier JSON produit : <nom_image>_output.json
    output_json_path = output_dir / f"{input_image_path.stem}_output.json"

    # Format JSON exigé par le système d'évaluation
    result_data = {
        "text": raw_text,
        "confidence": 0.95  # Champ optionnel
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()