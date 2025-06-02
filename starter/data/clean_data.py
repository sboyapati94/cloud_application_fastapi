import pandas as pd

df = pd.read_csv("starter/data/census.csv", skipinitialspace=True)
df.columns = df.columns.str.replace(" ", "")
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()
df.to_csv("starter/data/census_clean.csv", index=False)
