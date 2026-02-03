import pandas as pd
import numpy as np

def load_all_files():
    """
    Loads all 6 MIMIC-IV ED demo files with exact matching for your directory.
    """
    # Fixed filenames: 'edstays' (no underscore) and 'vitalsign' (no 's')
    paths = {
        'triage': 'data/raw/triage_demo.csv',
        'vitals': 'data/raw/vitalsign_demo.csv',
        'edstays': 'data/raw/edstays_demo.csv',
        'diagnosis': 'data/raw/diagnosis_demo.csv',
        'meds': 'data/raw/medrecon_demo.csv',
        'pyxis': 'data/raw/pyxis_demo.csv'
    }
    
    dfs = {}
    for name, path in paths.items():
        try:
            dfs[name] = pd.read_csv(path)
            print(f"✅ Loaded {name}: {dfs[name].shape}")
        except FileNotFoundError:
            print(f"❌ Error: Could not find {path}. Verify the file is in data/raw/")
            
    return dfs

def preprocess_master(dfs):
    """
    Merges all files and creates new features for Data Mining IE 7275.
    """
    # 1. Start with Triage and Vitals
    triage = dfs['triage']
    vitals = dfs['vitals']
    
    # FIX: Use 'acuity' (MIMIC column) instead of 'triage_level'
    if 'acuity' in triage.columns:
        triage['acuity'] = triage['acuity'].fillna(triage['acuity'].mode()[0])
    
    # 2. Clean Vitals
    numeric_vitals = vitals.select_dtypes(include=[np.number])
    vitals_cleaned = vitals.fillna(numeric_vitals.median())
    
    # 3. Master Merge (MIMIC-IV uses subject_id and stay_id)
    master = triage.merge(vitals_cleaned, on=['subject_id', 'stay_id'], how='left')

    # 4. Feature: Diagnosis Count (From diagnosis_demo.csv)
    if 'diagnosis' in dfs:
        diag_counts = dfs['diagnosis'].groupby('subject_id').size().reset_index(name='num_prior_cond')
        master = master.merge(diag_counts, on='subject_id', how='left')

    # 5. Feature: Home Med Count (From medrecon_demo.csv)
    if 'meds' in dfs:
        med_counts = dfs['meds'].groupby('subject_id').size().reset_index(name='num_home_meds')
        master = master.merge(med_counts, on='subject_id', how='left')

    # 6. Feature: ER Meds Administered (From pyxis_demo.csv)
    if 'pyxis' in dfs:
        pyxis_counts = dfs['pyxis'].groupby('stay_id').size().reset_index(name='er_meds_given')
        master = master.merge(pyxis_counts, on='stay_id', how='left')

    # 7. Add ED Stay details (From edstays_demo.csv)
    if 'edstays' in dfs:
        master = master.merge(dfs['edstays'], on=['subject_id', 'stay_id'], how='left')

    # Final cleanup: Replace NaN counts with 0
    fill_cols = ['num_prior_cond', 'num_home_meds', 'er_meds_given']
    for col in fill_cols:
        if col in master.columns:
            master[col] = master[col].fillna(0)
    
    return master

if __name__ == "__main__":
    # Step 1: Load everything
    all_data = load_all_files()
    
    if all_data:
        # Step 2: Process and Merge
        final_df = preprocess_master(all_data)
        
        # Step 3: Save the Master Dataset
        output_file = 'data/processed/master_dataset.csv'
        final_df.to_csv(output_file, index=False)
        
        print("\n--- EMERGEN AI: PREPROCESSING COMPLETE ---")
        print(f"Master file created at: {output_file}")
        print(f"Final dataset shape: {final_df.shape}")
        print("\nFirst 5 rows of merged data:")
        print(final_df.head())