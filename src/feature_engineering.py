import os
import pandas as pd
import numpy as np

def calculate_patient_features(df):
    """Calculates clinical indices from vitals using the merged column names"""
    # Using the '_y' suffix names seen in your master_dataset spreadsheet
    # Shock Index: HR / SBP
    df['shock_index'] = df['heartrate_y'] / df['sbp_y']
    
    # Pulse Pressure: SBP - DBP
    df['pulse_pressure'] = df['sbp_y'] - df['dbp_y']
    
    # Complexity Score: Number of conditions + home meds
    # These names likely stayed the same, but we use fillna to be safe
    df['complexity_score'] = df['num_prior_cond'].fillna(0) + df['num_home_meds'].fillna(0)
    
    return df

def calculate_context_features(df):
    """Calculates ER congestion/resources as per your flowchart"""
    # Using 'charttime' from your spreadsheet to calculate arrival patterns
    if 'charttime' in df.columns:
        df['charttime'] = pd.to_datetime(df['charttime'])
        df['arrival_hour'] = df['charttime'].dt.hour
        df['arrival_day'] = df['charttime'].dt.dayofweek
    return df

if __name__ == "__main__":
    # Load the master dataset
    df = pd.read_csv('data/processed/master_dataset.csv')
    
    # Run the pipeline
    df = calculate_patient_features(df)
    df = calculate_context_features(df)
    
    # Cleanup: Replace infinities from division and fill remaining NaNs
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # Drop datetime columns to speed up CSV writing (features already extracted)
    datetime_cols = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns
    if len(datetime_cols) > 0:
        df = df.drop(columns=list(datetime_cols))

    # Save the 'Model Ready' file
    output_path = os.getenv("MODEL_READY_PATH", "data/processed/model_ready.csv")
    output_format = os.getenv("MODEL_READY_FORMAT", "csv").lower()
    if output_format == "parquet":
        try:
            df.to_parquet(output_path, index=False)
            print(f"✅ Feature Engineering Complete. '{output_path}' is ready for training!")
        except Exception:
            output_format = "csv"

    if output_format == "csv":
        df.to_csv(
            output_path,
            index=False,
            date_format="%Y-%m-%d %H:%M:%S",
            chunksize=200_000,
        )
        print(f"✅ Feature Engineering Complete. '{output_path}' is ready for training!")
    print(f"Engineered columns: ['shock_index', 'pulse_pressure', 'arrival_hour']")
