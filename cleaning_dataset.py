import pandas as pd
df = pd.read_csv('results.csv')
df_cleaned = df[~(df == -10).any(axis=1)]
df_cleaned.to_csv('result_cleaned.csv', index=False)