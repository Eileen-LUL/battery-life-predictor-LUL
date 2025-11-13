import pandas as pd

def clean_data(df):
    df = df.rename(columns=lambda x: x.strip().lower())
    df = df.dropna()
    df = df[df["capacity"] > 0]   # Remove bad readings
    df = df[df["cycle"] >= 0]
    return df
