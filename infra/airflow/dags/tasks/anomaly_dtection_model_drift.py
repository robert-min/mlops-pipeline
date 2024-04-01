def is_model_drift():
    import glob
    import os
    import pandas as pd
    
    DRFIT_THRESHOLD = 30
    
    anomalies_files = glob.glob("dags/outputs/inference/*_anomalies.csv")
    anomalies_files.sort(key=os.path.getmtime)
    latest_anomalies_file = anomalies_files[-1]
    
    df = pd.read_csv(latest_anomalies_file)
    
    return len(df) > DRFIT_THRESHOLD
