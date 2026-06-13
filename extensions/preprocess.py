import argparse
import pandas as pd
import os
import time

# === DEFAULT CONFIGURATION ===
DEFAULT_SOURCE_CSV = ''
DEFAULT_TARGET_CSV = ''
DEFAULT_SOURCE_NAME = ''
DEFAULT_TARGET_NAME = ''
DEFAULT_K_CORE = 5
DEFAULT_OUT_DIR = 'dataset'
# =============================

def k_core_filter(df, k):
    """Iteratively filters until all users and items have at least k interactions."""
    iteration = 0
    prev_len = 0
    while len(df) != prev_len:
        prev_len = len(df)
        iteration += 1
        # Filter users with at least k interactions
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Filter items with at least k interactions
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df['item_id'].isin(valid_items)]
        print(f"      [Iter {iteration}] {len(df)} interactions remaining")
    return df

def main():
    parser = argparse.ArgumentParser(description="Preprocesses CSV files for RecBole-CDR (Cross-Domain Recommendation).")
    parser.add_argument('--source_csv', type=str, default=DEFAULT_SOURCE_CSV, help='Path to the source domain CSV')
    parser.add_argument('--target_csv', type=str, default=DEFAULT_TARGET_CSV, help='Path to the target domain CSV')
    parser.add_argument('--source_name', type=str, default=DEFAULT_SOURCE_NAME, help='Short name for source domain (e.g., AmazonCloth)')
    parser.add_argument('--target_name', type=str, default=DEFAULT_TARGET_NAME, help='Short name for target domain (e.g., AmazonSport)')
    parser.add_argument('--k_core', type=int, default=DEFAULT_K_CORE, help='Threshold for k-core filtering')
    parser.add_argument('--out_dir', type=str, default=DEFAULT_OUT_DIR, help='Base output directory')
    args = parser.parse_args()

    start_time = time.time()

    print(f"[1/5] Loading CSVs...")
    source_df = pd.read_csv(args.source_csv)
    target_df = pd.read_csv(args.target_csv)

    # Check if columns are named "parent_asin" and rename them to "item_id"
    if 'parent_asin' in source_df.columns:
        source_df = source_df.rename(columns={'parent_asin': 'item_id'})
    if 'parent_asin' in target_df.columns:
        target_df = target_df.rename(columns={'parent_asin': 'item_id'})

    # Check for the existence of minimum required columns
    for col in ['user_id', 'item_id', 'rating']:
        if col not in source_df.columns:
            raise ValueError(f"Missing column in source CSV: {col}")
        if col not in target_df.columns:
            raise ValueError(f"Missing column in target CSV: {col}")

    print(f"      {args.source_name}: {len(source_df)} interactions (original total)")
    print(f"      {args.target_name}: {len(target_df)} interactions (original total)")

    print(f"\n[2/5] Extracting common users (Step 1 from paper)...")
    common_users = set(source_df['user_id'].unique()) & set(target_df['user_id'].unique())
    source_df = source_df[source_df['user_id'].isin(common_users)].copy()
    target_df = target_df[target_df['user_id'].isin(common_users)].copy()
    print(f"      Common users found: {len(common_users)}")

    print(f"\n[3/5] Iterative K-core filtering on single domains (Step 2, 3, 4 from paper)...")
    print(f"   -> Filtering {args.source_name}:")
    source_df = k_core_filter(source_df, args.k_core)
    print(f"   -> Filtering {args.target_name}:")
    target_df = k_core_filter(target_df, args.k_core)

    print(f"\n[4/5] Re-extracting common users (Step 5 from paper)...")
    common_final = set(source_df['user_id'].unique()) & set(target_df['user_id'].unique())
    source_df = source_df[source_df['user_id'].isin(common_final)]
    target_df = target_df[target_df['user_id'].isin(common_final)]
    
    print(f"\n      === FINAL RESULT ===")
    print(f"      {args.source_name}: {len(source_df)} interactions, {source_df['user_id'].nunique()} users, {source_df['item_id'].nunique()} items")
    print(f"      {args.target_name}: {len(target_df)} interactions, {target_df['user_id'].nunique()} users, {target_df['item_id'].nunique()} items")
    print(f"      Final common users: {len(common_final)}")

    if len(common_final) < 10:
        print("\nWARNING: Too few common users! Consider lowering the k_core parameter.")

    print(f"\n[5/5] Saving .inter files in '{args.out_dir}'...")
    source_dataset_name = f"{args.source_name}_{args.target_name}_commonUser_{args.k_core}-core"
    target_dataset_name = f"{args.target_name}_{args.source_name}_commonUser_{args.k_core}-core"

    source_dir = os.path.join(args.out_dir, source_dataset_name)
    target_dir = os.path.join(args.out_dir, target_dataset_name)
    os.makedirs(source_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    source_out = source_df[['user_id', 'item_id', 'rating']]
    target_out = target_df[['user_id', 'item_id', 'rating']]

    source_path = os.path.join(source_dir, f"{source_dataset_name}.inter")
    target_path = os.path.join(target_dir, f"{target_dataset_name}.inter")

    source_out.to_csv(source_path, sep='\t', index=False, header=["user_id:token", "item_id:token", "rating:float"])
    target_out.to_csv(target_path, sep='\t', index=False, header=["user_id:token", "item_id:token", "rating:float"])

    print(f"Files successfully created in {time.time() - start_time:.1f} seconds:")
    print(f"  - {source_path}")
    print(f"  - {target_path}")

if __name__ == '__main__':
    main()
