import json
import torch
import clip
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np
from tqdm import tqdm
import os
import sys
from transformers import DistilBertTokenizer, DistilBertModel, CLIPTokenizer

# Add parent directory to path for importing modules
parent_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.insert(0, parent_dir)
# from tune.babel_humanml3d.model import ClipTextEncoder
from tune.custom_LaCLIP.model import ClipTextEncoder
from tune.babel_humanml3d.tokenizer import SimpleTokenizer

# Function to load the text encoder
def load_text_encoder(encoder_type, device):
    if encoder_type == "CLIP":
        model, _ = clip.load("ViT-B/32", device=device)
    elif encoder_type == "LaCLIP":
        clip_model, _ = clip.load("ViT-B/32", device=device)
        checkpoint = torch.load("/home/jiaxu/projects/sp-motionclip/tune/custom_LaCLIP/checkpoints/FinetuneLaCLIP_NoContrastive_DS_Custom_LR_5e-05_WD_0.01_EP_50_MG_2_NN_1_gki33zlw", map_location=device)
        clip_model.load_state_dict(checkpoint, strict=False)
        return clip_model, None
    elif encoder_type == "MotionLaCLIP":
        model = ClipTextEncoder(
            hf_pretrained_name="openai/clip-vit-base-patch32",
            pretrained_ckpt_path="/home/jiaxu/projects/sp-motionclip/tune/motionlaclip/BEST_dropout_coslr_FinetuneLaCLIP_DS_HumanML3D_LR_5e-05_WD_0.01_EP_16_y8ksv238/finetunelaclip_epoch_8.pth",
            dropout=0.1
        ).to(device)
    elif encoder_type == "MotionLaCLIP+":
        model_path = "/home/jiaxu/projects/sp-motionclip/tune/custom_LaCLIP/checkpoints/FinetuneLaCLIP_NoContrastive_CosSim_DS_CustomAll_LR_0.0001_WD_0.001_EP_150_MG_1_NN_6_yf6zuocb"
        tokenizer = CLIPTokenizer.from_pretrained(f"{model_path}/tokenizer")
        model = ClipTextEncoder(hf_pretrained_name="openai/clip-vit-base-patch32")
        state_dict = torch.load(f"{model_path}/model/pytorch_model.bin", map_location="cpu") 
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model, tokenizer 
    elif encoder_type == "DistilBERT":
        model_path = "/home/jiaxu/projects/sp-motionclip/tune/custom/checkpoints/FinetuneLaCLIP_Centroids_DS_Custom_LR_5e-05_WD_0.01_EP_5_1khj0wk8/"
        tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        model = DistilBertModel.from_pretrained(model_path).to(device)
        model.eval()
        return model, tokenizer  
    else:
        raise ValueError(f"Unsupported encoder type: {encoder_type}")
    
    return model  

# Function to compute embeddings
def compute_embeddings(model, tokenizer, texts, device, encoder_type, batch_size=16):
    embeddings = []
    if encoder_type in ["CLIP", "LaCLIP"]:
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing embeddings for {encoder_type}"):
                batch_texts = texts[i:i+batch_size]
                text_tokens = clip.tokenize(batch_texts, truncate=True).to(device)
                batch_embeddings = model.encode_text(text_tokens)
                embeddings.append(batch_embeddings.cpu().numpy())

    elif encoder_type == "MotionLaCLIP":
        model.eval()
        tokenizer = SimpleTokenizer()
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing embeddings for {encoder_type}"):
                batch_texts = texts[i:i+batch_size]
                encoded = [{'input_ids': tokenizer(text, context_length=60), 'attention_mask': (tokenizer(text, context_length=60) != 0).long()} for text in batch_texts]
                input_ids = torch.stack([item['input_ids'] for item in encoded]).to(device)
                attention_mask = torch.stack([item['attention_mask'] for item in encoded]).to(device)
                batch_embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings.append(batch_embeddings.cpu().numpy())

    elif encoder_type == "MotionLaCLIP+":
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing embeddings for {encoder_type}"):
                batch_texts = texts[i:i+batch_size]
                encoded = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                batch_embeddings = model(input_ids=input_ids, attention_mask=attention_mask)
                embeddings.append(batch_embeddings.cpu().numpy())

    elif encoder_type == "DistilBERT":
        model.eval()
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc=f"Computing embeddings for {encoder_type}"):
                batch_texts = texts[i:i+batch_size]
                encoded = tokenizer(batch_texts, padding="max_length", truncation=True, max_length=64, return_tensors="pt")
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                batch_embeddings = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
                embeddings.append(batch_embeddings.cpu().numpy())
    else:
        raise NotImplementedError(f"Embedding computation not implemented for {encoder_type}")

    return np.concatenate(embeddings, axis=0)

# Save embeddings to a JSON file
def save_embeddings_to_json(file_path, embeddings_data):
    with open(file_path, 'w') as f:
        json.dump(embeddings_data, f, indent=4)

# Main function to process annotations
def process_annotations(file_path, output_file, reduction_method):
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
            grouped_texts[key] = [key] + sentences
        else:
            print(f"Unexpected data format for key {key}, skipping.")
    
    if not grouped_texts:
        print("No valid texts found in the JSON file.")
        return

    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    encoder_types = ["MotionLaCLIP+"]
    embeddings_data = {}

    for encoder_type in tqdm(encoder_types, desc="Encoders"):
        print(f"\nProcessing embeddings for {encoder_type}...")
        embeddings_data[encoder_type] = {}
        try:
            model, tokenizer = load_text_encoder(encoder_type, device)  
            model.eval()
            for action, texts in grouped_texts.items():
                embeddings = compute_embeddings(model, tokenizer, texts, device, encoder_type)

                # Choose dimensionality reduction method
                if reduction_method == "PCA":
                    reducer = PCA(n_components=2)
                elif reduction_method == "tSNE":
                    reducer = TSNE(n_components=2)
                else:
                    raise ValueError(f"Invalid reduction method: {reduction_method}. Choose 'PCA' or 'tSNE'.")

                embeddings_2d = reducer.fit_transform(embeddings)
                embeddings_data[encoder_type][action] = {
                    "embeddings_2d": embeddings_2d.tolist()
                }
                print(f"Processed {len(texts)} texts for action '{action}' with {encoder_type} using {reduction_method}")
        except Exception as e:
            print(f"Error with encoder {encoder_type}: {e}")
            continue

    # Save the embeddings data to JSON
    save_embeddings_to_json(output_file, embeddings_data)
    print(f"Embeddings saved to {output_file}")

# Execute the script
if __name__ == "__main__":
    file_path = "/home/jiaxu/projects/sp-motionclip/dataset/custom/custom_2000_test_for_plot.json"  # Replace with your actual JSON file path
    output_file = "embeddings_custom_2000_test.json"  # Output JSON file path
    reduction_method = "PCA"
    process_annotations(file_path, output_file, reduction_method)
