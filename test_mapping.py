import argparse
from recbole_cdr.quick_start import run_recbole_cdr
import yaml

# A mock run script to just load the dataset and test the token mapping
from recbole_cdr.config import CDRConfig
from recbole_cdr.data import create_dataset

def test():
    overall_config = 'recbole_cdr/properties/overall.yaml'
    data_config = 'recbole_cdr/properties/dataset/AmazonKindle_AmazonBooks_commonUser_5-core.yaml'
    model_config = 'recbole_cdr/properties/model/DGCDR.yaml'
    
    config = CDRConfig(model='DGCDR', config_file_list=[overall_config, data_config, model_config])
    dataset = create_dataset(config)
    
    print(f"Source items: {dataset.source_domain_dataset.item_num}")
    print(f"Target items: {dataset.target_domain_dataset.item_num}")
    print(f"Total items (DGCDR logic): {dataset.target_domain_dataset.item_num + dataset.source_domain_dataset.item_num - dataset.num_overlap_item}")
    
    # Try mapping ID 1
    try:
        source_token = dataset.source_domain_dataset.id2token(dataset.source_domain_dataset.iid_field, 1)
        print(f"Source ID 1 -> Token: {source_token}")
    except Exception as e:
        print(f"Source token mapping error: {e}")

if __name__ == "__main__":
    test()
