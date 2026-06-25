import json
import csv
import os
# pyrefly: ignore [missing-import]
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

def load_valid_ids(csv_path):
    print(f"Loading IDs from {os.path.basename(csv_path)}...")
    valid_ids = set()
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        asin_index = header.index('parent_asin')
        for row in reader:
            valid_ids.add(row[asin_index])
    return valid_ids

def create_text_document(item):
    parts = []
    
    title = item.get('title')
    if title:
        parts.append(f"Title: {title}")
        
    categories = item.get('categories')
    if categories and isinstance(categories, list):
        parts.append(f"Categories: {', '.join(categories)}")
        
    features = item.get('features')
    if features and isinstance(features, list):
        parts.append(f"Features: {'. '.join(features)}")
        
    description = item.get('description')
    if description and isinstance(description, list):
        parts.append(f"Description: {' '.join(description)}")
        
    return " | ".join(parts)

def process_dataset(domain_name, csv_path, jsonl_path, output_dir, model, batch_size=512):
    valid_ids = load_valid_ids(csv_path)
    
    print(f"\nProcessing {domain_name} metadata...")
    item_texts = {}
    
    # Read and parse valid items
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Parsing JSONL ({domain_name})"):
            item = json.loads(line)
            item_id = item.get('parent_asin')
            
            if item_id in valid_ids:
                doc = create_text_document(item)
                item_texts[item_id] = doc
                
    # We should have exactly len(valid_ids) items
    assert len(item_texts) == len(valid_ids), f"Mismatch: expected {len(valid_ids)} items, found {len(item_texts)}"
    
    print(f"Extracting embeddings for {len(item_texts)} items (Batch size: {batch_size})...")
    
    # Prepare data for batching
    ids = list(item_texts.keys())
    texts = [item_texts[i] for i in ids]
    
    # Generate embeddings
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True, device=device)
    
    # Move back to CPU to save
    embeddings = embeddings.cpu()
    
    # Create the dictionary mapping ID -> Tensor
    embeddings_dict = {item_id: embeddings[i] for i, item_id in enumerate(ids)}
    
    # Save the dictionary as a .pt file
    output_path = os.path.join(output_dir, f"text_embeddings_{domain_name}.pt")
    torch.save(embeddings_dict, output_path)
    print(f"Saved successfully to {output_path}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    csv_dir = os.path.join(base_dir, "csv_datasets")
    meta_dir = os.path.join(base_dir, "item_metadata")
    ext_dir = os.path.join(os.path.dirname(__file__), "embeddings")
    
    os.makedirs(ext_dir, exist_ok=True)
    
    # Load the model
    print("Loading sentence-transformers model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Datasets to process
    datasets = [
        ("Musical_Instruments", "Musical_Instruments.csv", "meta_Musical_Instruments.jsonl"),
        ("CDs_and_Vinyl", "CDs_and_Vinyl.csv", "meta_CDs_and_Vinyl.jsonl")
    ]
    
    for domain_name, csv_file, jsonl_file in datasets:
        csv_path = os.path.join(csv_dir, csv_file)
        jsonl_path = os.path.join(meta_dir, jsonl_file)
        
        process_dataset(domain_name, csv_path, jsonl_path, ext_dir, model, batch_size=512)

if __name__ == "__main__":
    main()
