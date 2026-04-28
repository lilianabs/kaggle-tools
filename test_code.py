import pandas as pd
from src.eda import plot_numerical_distributions, compute_numerical_summary, plot_categorical_distributions, compute_categorical_summary
from src.preprocessing import get_numerical_and_categorical_features

df = pd.read_csv('data/penguins.csv')

num_feats, cat_feats = get_numerical_and_categorical_features(df)

print(f"""Numerical features: {num_feats}\n"f"Categorical features: {cat_feats}""")

plot_numerical_distributions(df, num_feats, bins=30)
summary = compute_numerical_summary(df, num_feats, cardinality_threshold=10)
print(summary)

plot_categorical_distributions(df, cat_feats)
summary = compute_categorical_summary(df, cat_feats, cardinality_threshold=10)
print(summary)
