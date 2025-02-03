import json
import torch
import clip
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os
import sys

# Add parent directory to path for importing modules
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
from tune.babel_humanml3d.model import ClipTextEncoder
from tune.babel_humanml3d.tokenizer import SimpleTokenizer

# Function to load the text encoder
def load_text_encoder(encoder_type, device):
    if encoder_type == "CLIP":
        model, _ = clip.load("ViT-B/32", device=device)
    elif encoder_type == "LaCLIP":
        model, _ = clip.load("ViT-B/32", device=device)
        checkpoint = torch.load("/home/jiaxu/projects/sp-motionclip/tune/laclip/laion400m_laclip.pt", map_location=device)
        model.load_state_dict(checkpoint.get("state_dict", checkpoint), strict=False)
    elif encoder_type == "MotionLaCLIP":
        model = ClipTextEncoder(
            hf_pretrained_name="openai/clip-vit-base-patch32",
            pretrained_ckpt_path="/home/jiaxu/projects/sp-motionclip/tune/motionlaclip/BEST_dropout_coslr_FinetuneLaCLIP_DS_HumanML3D_LR_5e-05_WD_0.01_EP_16_y8ksv238/finetunelaclip_epoch_8.pth",
            dropout=0.1
        ).to(device)
    elif encoder_type == "MotionLaCLIP+":
        model = ClipTextEncoder(
            hf_pretrained_name="openai/clip-vit-base-patch32",
            pretrained_ckpt_path="/home/jiaxu/projects/sp-motionclip/tune/motionlaclipplus/latest/finetunelaclip_epoch_5.pth",
            dropout=0.1
        ).to(device)
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
    return model

# Function to compute embeddings
def compute_embeddings(encoder, texts, device, encoder_type, batch_size=16):
    embeddings = []
    if encoder_type in ["CLIP", "LaCLIP"]:
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing embeddings for {encoder_type}"):
                batch_texts = texts[i:i+batch_size]
                text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)
                batch_embeddings = encoder.encode_text(text_tokens)
                embeddings.append(batch_embeddings.cpu().numpy())
    elif encoder_type in ["MotionLaCLIP", "MotionLaCLIP+"]:
        encoder.eval()
        tokenizer = SimpleTokenizer()
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing embeddings for {encoder_type}"):
                batch_texts = texts[i:i+batch_size]
                encoded = [{'input_ids': tokenizer(text, context_length=60), 'attention_mask': (tokenizer(text, context_length=60) != 0).long()} for text in batch_texts]
                input_ids = torch.stack([item['input_ids'] for item in encoded]).to(device)
                attention_mask = torch.stack([item['attention_mask'] for item in encoded]).to(device)
                batch_embeddings = encoder(input_ids=input_ids, attention_mask=attention_mask)
                embeddings.append(batch_embeddings.cpu().numpy())
    else:
        raise NotImplementedError(f"Embedding computation not implemented for {encoder_type}")
    return np.concatenate(embeddings, axis=0)

# Save embeddings to a JSON file
def save_embeddings_to_json(file_path, embeddings_data):
    with open(file_path, 'w') as f:
        json.dump(embeddings_data, f, indent=4)

# Main function to process annotations
def process_annotations(file_path, output_file):
    # Load the JSON file
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return
    
    # Prepare texts and groups
    grouped_texts = {}
    for key, sentences in data.items():
        if isinstance(sentences, list):
            grouped_texts[key] = sentences
        else:
            print(f"Unexpected data format for key {key}, skipping.")
    
    if not grouped_texts:
        print("No valid texts found in the JSON file.")
        return

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    encoder_types = ["CLIP", "LaCLIP", "MotionLaCLIP", "MotionLaCLIP+"]
    embeddings_data = {}

    for encoder_type in tqdm(encoder_types, desc="Encoders"):
        print(f"\nProcessing embeddings for {encoder_type}...")
        embeddings_data[encoder_type] = {}
        try:
            encoder = load_text_encoder(encoder_type, device)
            for action, texts in grouped_texts.items():
                embeddings = compute_embeddings(encoder, texts, device, encoder_type)
                pca = PCA(n_components=2)
                embeddings_2d = pca.fit_transform(embeddings)
                embeddings_data[encoder_type][action] = {
                    "embeddings_2d": embeddings_2d.tolist()
                }
                print(f"Processed {len(texts)} texts for action '{action}' with {encoder_type}")
        except Exception as e:
            print(f"Error with encoder {encoder_type}: {e}")
            continue

    # Save the embeddings data to JSON
    save_embeddings_to_json(output_file, embeddings_data)
    print(f"Embeddings saved to {output_file}")

# Execute the script
if __name__ == "__main__":
    file_path = "eval.json"  # Replace with your actual JSON file path
    output_file = "embeddings_by_model.json"  # Output JSON file path
    process_annotations(file_path, output_file)
