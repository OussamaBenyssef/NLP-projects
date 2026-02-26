import os
import sys
import pandas as pd
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_config, setup_logger

def prep_atlaset():
    config = load_config()
    logger = setup_logger("prep_atlaset", log_level=config["project"]["log_level"])
    
    atlaset_dir = os.path.dirname(config["paths"]["data"]["atlaset"])
    atlaset_in_dir = os.path.join(atlaset_dir, "atlaset")
    out_parquet = config["paths"]["data"]["atlaset"]
    
    parquet_files = glob.glob(os.path.join(atlaset_in_dir, "*.parquet"))
    
    if not parquet_files:
        logger.error(f"No parquet files found in {atlaset_in_dir}")
        return
        
    dfs = []
    for f in parquet_files:
        df = pd.read_parquet(f)
        dfs.append(df)
        
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all.to_parquet(out_parquet, index=False)
        logger.info(f"Atlaset prepared and saved to {out_parquet} with {len(df_all)} records.")
    else:
        logger.error("Failed to load Atlaset parquets.")

if __name__ == "__main__":
    prep_atlaset()
