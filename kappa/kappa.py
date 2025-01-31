import pandas as pd
from sklearn.metrics import cohen_kappa_score

# Load the data
df = pd.read_csv('ratings.csv')

# Compute Cohen's kappa between all columns
columns = df.columns
kappa_matrix = pd.DataFrame(index=columns, columns=columns)

for col1 in columns:
    for col2 in columns:
        if col1 != col2:
            kappa_matrix.loc[col1, col2] = str("%.2f" % cohen_kappa_score(df[col1], df[col2]))
        else:
            kappa_matrix.loc[col1, col2] = 1.0  # Kappa with itself is always 1

print(kappa_matrix)