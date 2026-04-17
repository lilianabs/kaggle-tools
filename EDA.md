# Key Strategies for Effective EDA

1. **Understand Data Structure:** Quickly check data types, shapes, and summary statistics using `df.info()`, `df.describe()`, and `df.head()` to understand the data's scale and types (numerical, categorical).
1. **Target Variable Analysis:** Analyze the target variable (distribution for regression, class balance for classification) as it directly impacts your choice of loss functions and evaluation metrics.
1. **Univariate Analysis:** Visualize individual feature distributions using histograms, density plots, and box plots to identify skewness and outliers.
1. **Bivariate/Multivariate Analysis:** Use scatter plots and pair plots to explore relationships between predictors (X) and the target (Y). Utilize heatmaps to detect correlations among predictors to identify potential multicollinearity.
1. **Handle Missing Data:** Identify the percentage of missing values per column and visualize their distribution. Decide on imputation strategies (mean, median, mode, or specialized imputation) based on data insights.
1. **Outlier Detection:** Use box plots to identify outliers, and evaluate their impact, deciding whether to cap, transform, or remove them, as they can heavily influence models.
1. **Identify Data Leakage:** Crucially, check if any feature already contains information about the target variable that wouldn't be available at prediction time.
1. **Create Hypotheses:** Document findings and form hypotheses about which features will be most predictive.
