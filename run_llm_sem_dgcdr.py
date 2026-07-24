import argparse
import os
from recbole_cdr.quick_start import run_recbole_cdr

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run LLM-SemDGCDR (Solution A: Semantic Supervision for Disentanglement)")
    parser.add_argument('--model', '-m', type=str, default='DGCDR', help='Name of model')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs (overrides config if specified)')
    parser.add_argument('--dataset', type=str, default='AmazonCDs_AmazonInstruments_commonUser_3-core', 
                        help='Dataset config name')
    parser.add_argument('--semantic_loss_weight', type=float, default=0.1, help='Weight for semantic disentanglement loss')
    
    args, _ = parser.parse_known_args()
    
    overall_config = 'recbole_cdr/properties/overall.yaml'
    data_config = f'recbole_cdr/properties/dataset/{args.dataset}.yaml'
    if not os.path.exists(data_config):
        # Fallback to CDs/Instruments dataset if specified path does not exist
        data_config = 'recbole_cdr/properties/dataset/AmazonCDs_AmazonInstruments_commonUser_3-core.yaml'
        
    model_config = f'recbole_cdr/properties/model/{args.model}.yaml'
    
    config_file_list = [overall_config, data_config, model_config]
    
    base_dir = os.path.abspath(os.path.dirname(__file__))
    ext_dir = os.path.join(base_dir, 'extensions', 'offline_extraction', 'embeddings')
    
    src_shared_path = os.path.join(ext_dir, 'shared_text_embeddings_CDs_and_Vinyl.pt')
    src_spec_path = os.path.join(ext_dir, 'specific_text_embeddings_CDs_and_Vinyl.pt')
    tgt_shared_path = os.path.join(ext_dir, 'shared_text_embeddings_Musical_Instruments.pt')
    tgt_spec_path = os.path.join(ext_dir, 'specific_text_embeddings_Musical_Instruments.pt')
    
    config_dict = {
        'use_semantic_disentanglement': True,
        'semantic_loss_weight': args.semantic_loss_weight,
        'source_shared_text_path': src_shared_path,
        'source_specific_text_path': src_spec_path,
        'target_shared_text_path': tgt_shared_path,
        'target_specific_text_path': tgt_spec_path,
    }
    
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs
        
    print("=" * 70)
    print("      RUNNING LLM-SemDGCDR (Solution A: Semantic Supervision)")
    print("=" * 70)
    print(f"Dataset config          : {data_config}")
    print(f"Semantic Loss Weight    : {args.semantic_loss_weight}")
    print(f"Source Shared Text Path : {src_shared_path}")
    print(f"Source Spec Text Path   : {src_spec_path}")
    print(f"Target Shared Text Path : {tgt_shared_path}")
    print(f"Target Spec Text Path   : {tgt_spec_path}")
    print("=" * 70)
    
    run_recbole_cdr(model=args.model, config_file_list=config_file_list, config_dict=config_dict)
