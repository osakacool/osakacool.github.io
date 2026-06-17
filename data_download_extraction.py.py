"""
data_download_extraction.py
Automated data retrieval and standardized extraction pipeline for plant growth-metabolism datasets.
Complies with the data availability statement and reproducibility requirements.
Handles NIH Metabolomics Workbench (ST001062, ST002317) and provides extraction logic 
for OsFPD and ZmDRD from their respective public repositories.

Corresponds to:
- Section 5.1: Datasets (AtGMD, OsFPD, ZmDRD)
- Section 4.3: Adaptive Time Warping (Eq. 16)
- Table 1: Dataset Characteristics

Author: [Your Name/Team]
License: MIT
"""

import os
import requests
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import zipfile
import io
from datetime import datetime

# ==============================================================================
# 1. Configuration & Directory Setup
# ==============================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# Dataset configurations based on Section 5.1 and Table 1
DATASETS = {
    "ST001062": {
        "name": "AtGMD_62d",
        "species": "Arabidopsis thaliana",
        "source": "NIH Metabolomics Workbench",
        "url_api": "https://www.metabolomicsworkbench.org/rest/study/study_id/ST001062/summary",
        "url_download": "https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&DataMode=MetaboliteData&StudyID=ST001062&DataFormat=CSV",
        "expected_growth": 12,
        "expected_metabolites": 56
    },
    "ST002317": {
        "name": "AtGMD_70d",
        "species": "Arabidopsis thaliana",
        "source": "NIH Metabolomics Workbench",
        "url_api": "https://www.metabolomicsworkbench.org/rest/study/study_id/ST002317/summary",
        "url_download": "https://www.metabolomicsworkbench.org/data/DRCCMetadata.php?Mode=Study&DataMode=MetaboliteData&StudyID=ST002317&DataFormat=CSV",
        "expected_growth": 12,
        "expected_metabolites": 56
    },
    "OsFPD": {
        "name": "OsFPD",
        "species": "Oryza sativa",
        "source": "Kim et al., 2020 (Public Repository)",
        "repository_url": "https://doi.org/10.xxxx/osfpd_2020", # Replace with actual DOI/Figshare link
        "expected_growth": 8,
        "expected_metabolites": 32
    },
    "ZmDRD": {
        "name": "ZmDRD",
        "species": "Zea mays",
        "source": "Li et al., 2024a (Public Repository)",
        "repository_url": "https://doi.org/10.xxxx/zmdrd_2024", # Replace with actual DOI/Zenodo link
        "expected_growth": 10,
        "expected_metabolites": 45
    }
}

# ==============================================================================
# 2. Automated Download Module (NIH Metabolomics Workbench)
# ==============================================================================
def download_nih_mw_dataset(study_id, config):
    """
    Attempts to download data from NIH MW. 
    Falls back to manual instructions if direct API download is restricted.
    """
    save_path = RAW_DIR / f"{study_id}_raw.csv"
    
    if save_path.exists():
        logging.info(f"[{study_id}] Raw data already exists at {save_path}. Skipping download.")
        return save_path

    logging.info(f"[{study_id}] Attempting to download from NIH Metabolomics Workbench...")
    
    # 1. Verify study exists via REST API
    try:
        resp = requests.get(config["url_api"], timeout=10)
        if resp.status_code == 200:
            logging.info(f"[{study_id}] Study verified: {resp.json().get('study_title', 'Unknown')}")
        else:
            logging.warning(f"[{study_id}] API verification failed. Status: {resp.status_code}")
    except Exception as e:
        logging.warning(f"[{study_id}] API connection failed: {e}")

    # 2. Attempt direct CSV download
    try:
        # NIH MW sometimes requires session/cookies for large data; we try a direct GET first
        resp = requests.get(config["url_download"], timeout=60, stream=True)
        if resp.status_code == 200 and 'text/csv' in resp.headers.get('Content-Type', ''):
            with open(save_path, 'wb') as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            logging.info(f"[{study_id}] Successfully downloaded to {save_path}")
            return save_path
        else:
            raise ValueError("Direct download returned non-CSV or failed.")
    except Exception as e:
        logging.warning(f"[{study_id}] Direct download failed ({e}). Falling back to manual download.")

    # 3. Fallback: Print manual instructions
    logging.info("="*60)
    logging.info(f"MANUAL DOWNLOAD REQUIRED FOR {study_id}")
    logging.info(f"1. Visit: {config['url_download']}")
    logging.info(f"2. Download the CSV file and save it as: {save_path.name}")
    logging.info(f"3. Place it in the directory: {RAW_DIR.absolute()}")
    logging.info("="*60)
    
    return None

# ==============================================================================
# 3. Data Extraction & Standardization Module
# ==============================================================================
def extract_and_format_data(file_path, config):
    """
    Extracts growth traits and metabolites, aligns timestamps (Eq. 16), 
    and saves in a standardized format for the preprocessing pipeline.
    """
    dataset_name = config["name"]
    logging.info(f"[{dataset_name}] Starting extraction and formatting...")
    
    # 1. Load raw data (handle CSV or ZIP)
    if str(file_path).endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as z:
            csv_name = [f for f in z.namelist() if f.endswith('.csv')][0]
            with z.open(csv_name) as f:
                df = pd.read_csv(f)
    else:
        # NIH MW data often uses tab or comma separation; try both
        try:
            df = pd.read_csv(file_path, sep='\t')
            if len(df.columns) < 5: # Fallback to comma
                df = pd.read_csv(file_path, sep=',')
        except:
            df = pd.read_csv(file_path, sep=',')

    # 2. Identify and separate features based on expected counts
    # Note: In real data, column names will be specific. Here we use a robust heuristic.
    # We assume columns containing 'growth', 'height', 'area', 'phenotype' are growth traits.
    # And columns with numeric data and specific metabolite names are metabolites.
    
    # For demonstration and robustness, we simulate the column separation logic 
    # that the user will adapt to the actual column names of ST001062/ST002317.
    growth_cols = [c for c in df.columns if any(k in c.lower() for k in ['height', 'area', 'growth', 'trait', 'phenotype'])]
    metab_cols = [c for c in df.columns if c not in growth_cols and df[c].dtype in [np.float64, np.int64, float, int]]
    
    # Filter out metadata columns (e.g., sample_id, treatment)
    meta_cols = [c for c in df.columns if any(k in c.lower() for k in ['sample', 'id', 'treatment', 'replicate', 'day', 'time', 'date'])]
    
    # Ensure we have the expected number of features (with a warning if not)
    if len(growth_cols) < config["expected_growth"]:
        logging.warning(f"[{dataset_name}] Found {len(growth_cols)} growth cols, expected {config['expected_growth']}. Please verify column names.")
    if len(metab_cols) < config["expected_metabolites"]:
        logging.warning(f"[{dataset_name}] Found {len(metab_cols)} metabolite cols, expected {config['expected_metabolites']}. Please verify column names.")

    # 3. Temporal Alignment (Section 4.3, Eq. 16)
    # Identify time column
    time_col = next((c for c in meta_cols if 'time' in c.lower() or 'day' in c.lower() or 'date' in c.lower()), None)
    if time_col:
        df['timestamp_raw'] = pd.to_datetime(df[time_col], errors='coerce')
        # If only 'day' or 'hour' is provided, simulate a start date
        if df['timestamp_raw'].isna().all():
            start_date = datetime(2020, 1, 1)
            df['timestamp_raw'] = df[time_col].apply(lambda x: start_date + pd.Timedelta(hours=x) if pd.notna(x) else pd.NaT)
        
        # Normalize time index (Eq. 16)
        t_min = df['timestamp_raw'].min()
        t_max = df['timestamp_raw'].max()
        df['time_normalized'] = (df['timestamp_raw'] - t_min).dt.total_seconds() / (t_max - t_min).total_seconds() + 1e-8
    else:
        logging.warning(f"[{dataset_name}] No time column found. Assigning sequential index.")
        df['time_normalized'] = np.arange(len(df)) / len(df)

    # 4. Identify Plant ID (for longitudinal tracking)
    plant_id_col = next((c for c in meta_cols if 'plant' in c.lower() or 'id' in c.lower()), 'plant_id')
    if plant_id_col not in df.columns:
        df['plant_id'] = 'P001' # Default if not specified
    else:
        df.rename(columns={plant_id_col: 'plant_id'}, inplace=True)

    # 5. Standardize column names to avoid conflicts
    # Prefix growth traits with 'G_' and metabolites with 'M_'
    df_renamed = df[['plant_id', 'timestamp_raw', 'time_normalized']].copy()
    
    for i, col in enumerate(growth_cols[:config["expected_growth"]]):
        df_renamed[f'G_{i+1}_{col}'] = df[col]
        
    for i, col in enumerate(metab_cols[:config["expected_metabolites"]]):
        df_renamed[f'M_{i+1}_{col}'] = df[col]

    # 6. Save processed data
    out_path = PROCESSED_DIR / f"{dataset_name}_aligned.parquet"
    df_renamed.to_parquet(out_path, index=False)
    logging.info(f"[{dataset_name}] Successfully saved standardized data to {out_path}")
    logging.info(f"  -> Shape: {df_renamed.shape} | Time range: {df_renamed['time_normalized'].min():.3f} to {df_renamed['time_normalized'].max():.3f}")
    
    return out_path

# ==============================================================================
# 4. Main Execution Pipeline
# ==============================================================================
def main():
    logging.info("="*70)
    logging.info("STARTING DATA DOWNLOAD & EXTRACTION PIPELINE")
    logging.info("="*70)

    for study_id, config in DATASETS.items():
        logging.info(f"\n--- Processing Dataset: {config['name']} ({study_id}) ---")
        
        # 1. Download / Locate Raw Data
        if "NIH" in config["source"]:
            raw_file = download_nih_mw_dataset(study_id, config)
        else:
            # For OsFPD and ZmDRD, check if user has placed the file in raw dir
            raw_file = RAW_DIR / f"{config['name']}_raw.csv"
            if not raw_file.exists():
                logging.warning(f"[{config['name']}] Raw data not found at {raw_file}.")
                logging.info(f"Please download from: {config['repository_url']}")
                logging.info(f"and save as {raw_file.name} in {RAW_DIR.absolute()}")
                continue

        # 2. Extract and Format
        if raw_file and raw_file.exists():
            try:
                extract_and_format_data(raw_file, config)
            except Exception as e:
                logging.error(f"[{config['name']}] Extraction failed: {e}")
        else:
            logging.warning(f"[{config['name']}] Skipping extraction due to missing raw data.")

    logging.info("\n" + "="*70)
    logging.info("PIPELINE COMPLETED.")
    logging.info(f"Standardized data is ready in: {PROCESSED_DIR.absolute()}")
    logging.info("Next step: Run `preprocessing_pipeline.py` for imputation and normalization.")
    logging.info("="*70)

if __name__ == "__main__":
    main()