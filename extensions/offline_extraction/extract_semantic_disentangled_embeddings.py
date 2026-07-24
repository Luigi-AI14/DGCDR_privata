import json
import csv
import os
import argparse
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

def create_shared_semantic_document(item, domain_name):
    """
    Extracts cross-domain transferable semantic features:
    Genre, general category, target audience, mood, themes.
    """
    parts = []
    
    title = item.get('title', '')
    if title:
        parts.append(f"Title: {title}")
        
    categories = item.get('categories', [])
    if categories and isinstance(categories, list):
        # Extract general high-level categories (transferable)
        parts.append(f"Broad Theme & Genre: {', '.join(categories[:3])}")
        
    description = item.get('description', [])
    if description and isinstance(description, list):
        # Take first part of description which usually contains overarching concept/theme
        short_desc = ' '.join(description)[:300]
        parts.append(f"Thematic Concept: {short_desc}")
        
    return " | ".join(parts) if parts else title

def create_specific_semantic_document(item, domain_name):
    """
    Extracts domain-specific features:
    Domain-specific attributes, brand/artist, technical specs, format, materials, etc.
    """
    parts = [f"Domain: {domain_name}"]
    
    features = item.get('features', [])
    if features and isinstance(features, list):
        parts.append(f"Technical Specs & Attributes: {'. '.join(features)}")
        
    categories = item.get('categories', [])
    if len(categories) > 3:
        parts.append(f"Specific Sub-categories: {', '.join(categories[3:])}")
        
    details = item.get('details', {})
    if isinstance(details, dict) and details:
        spec_str = ", ".join([f"{k}: {v}" for k, v in list(details.items())[:5]])
        parts.append(f"Item Details: {spec_str}")
        
    return " | ".join(parts)

def process_semantic_dataset(domain_name, csv_path, jsonl_path, output_dir, model, batch_size=256):
    valid_ids = load_valid_ids(csv_path)
    
    print(f"\n[Semantic Profiler] Processing {domain_name} metadata...")
    shared_texts = {}
    specific_texts = {}
    
    # Read and parse valid items
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f"Parsing JSONL ({domain_name})"):
            item = json.loads(line)
            item_id = item.get('parent_asin')
            
            if item_id in valid_ids:
                shared_texts[item_id] = create_shared_semantic_document(item, domain_name)
                specific_texts[item_id] = create_specific_semantic_document(item, domain_name)
                
    print(f"Extracted {len(shared_texts)} shared & specific text pairs.")
    
    ids = list(shared_texts.keys())
    s_texts = [shared_texts[i] for i in ids]
    p_texts = [specific_texts[i] for i in ids]
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Encoding text representations on {device} (batch size={batch_size})...")
    
    print("-> Encoding SHARED semantic representations (z_shared)...")
    shared_embeddings = model.encode(s_texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True, device=device).cpu()
    
    print("-> Encoding DOMAIN-SPECIFIC semantic representations (z_specific)...")
    specific_embeddings = model.encode(p_texts, batch_size=batch_size, show_progress_bar=True, convert_to_tensor=True, device=device).cpu()
    
    shared_dict = {item_id: shared_embeddings[i] for i, item_id in enumerate(ids)}
    specific_dict = {item_id: specific_embeddings[i] for i, item_id in enumerate(ids)}
    
    shared_out = os.path.join(output_dir, f"shared_text_embeddings_{domain_name}.pt")
    specific_out = os.path.join(output_dir, f"specific_text_embeddings_{domain_name}.pt")
    
    torch.save(shared_dict, shared_out)
    torch.save(specific_dict, specific_out)
    
    print(f"Saved shared embeddings to:    {shared_out}")
    print(f"Saved specific embeddings to:  {specific_out}")

def main():
    parser = argparse.ArgumentParser(description="Extract Dual Shared & Specific Semantic Embeddings (LLM-SemDGCDR)")
    parser.add_argument('--model_name', type=str, default='all-mpnet-base-v2', help="Model name for text encoding")
    parser.add_argument('--batch_size', type=int, default=256, help="Batch size for encoding")
    args = parser.parse_args()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    csv_dir = os.path.join(base_dir, "csv_datasets")
    meta_dir = os.path.join(base_dir, "item_metadata")
    ext_dir = os.path.join(os.path.dirname(__file__), "embeddings")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        print(f"Using GPU Acceleration: {torch.cuda.get_device_name(0)}")
    else:
        print("Warning: CUDA not available, falling back to CPU.")
        
    print(f"Loading embedding model: {args.model_name} on {device}...")
    model = SentenceTransformer(args.model_name, device=device)
    
    datasets = [
        ("Musical_Instruments", "Musical_Instruments.csv", "meta_Musical_Instruments.jsonl"),
        ("CDs_and_Vinyl", "CDs_and_Vinyl.csv", "meta_CDs_and_Vinyl.jsonl")
    ]
    
    for domain_name, csv_file, jsonl_file in datasets:
        csv_path = os.path.join(csv_dir, csv_file)
        jsonl_path = os.path.join(meta_dir, jsonl_file)
        
        if os.path.exists(csv_path) and os.path.exists(jsonl_path):
            process_semantic_dataset(domain_name, csv_path, jsonl_path, ext_dir, model, batch_size=args.batch_size)
        else:
            print(f"Skipping {domain_name}: {csv_path} or {jsonl_path} not found.")

if __name__ == "__main__":
    main()
