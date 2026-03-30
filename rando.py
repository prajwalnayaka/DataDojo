import pandas as pd
ds=pd.read_csv(r"D:\DataDojo\Industrial\iot_telemetry_data.csv")
mapping={True:"yes",False:"no"}
ds['motion']=ds['motion'].map(mapping)
ds.to_csv(r"D:\DataDojo\Industrial\iot_telemetry_data.csv")
