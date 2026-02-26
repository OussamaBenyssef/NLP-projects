import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger

def prep_nadi():
    config = load_config()
    logger = setup_logger("prep_nadi", log_level=config["project"]["log_level"])
    
    nadi_dir = config["paths"]["data"]["nadi"]
    train_tsv = os.path.join(nadi_dir, "NADI2022-Train", "Subtask1", "NADI2022_Subtask1_TRAIN.tsv")
    dev_tsv = os.path.join(nadi_dir, "NADI2022-Train", "Subtask1", "NADI2022_Subtask1_DEV.tsv")
    out_csv = os.path.join(nadi_dir, "nadi_texts.csv")
    
    dfs = []
    
    if os.path.exists(train_tsv):
        df_train = pd.read_csv(train_tsv, sep='\t')
        dfs.append(df_train)
    else:
        logger.warning(f"Train TSV not found at {train_tsv}")
        
    if os.path.exists(dev_tsv):
        df_dev = pd.read_csv(dev_tsv, sep='\t')
        dfs.append(df_dev)
    else:
        logger.warning(f"Dev TSV not found at {dev_tsv}")
        
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        # Rename columns to match pipeline expectations: id, text, label
        # NADI columns: #1_id, #2_content, #3_label
        df_all = df_all.rename(columns={
            "#1_id": "id",
            "#2_content": "text_raw",
            "#3_label": "label_original"
        })
        
        df_all.to_csv(out_csv, index=False)
        logger.info(f"NADI prepared and saved to {out_csv} with {len(df_all)} records.")
    else:
        logger.error("No NADI files found to process.")

if __name__ == "__main__":
    prep_nadi()
