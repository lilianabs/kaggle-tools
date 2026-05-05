import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_categorical_distribution(df: pd.DataFrame, column: str, top_n: int = 10) -> None:
    """Visualize the distribution of a categorical variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the categorical column to visualize.
        top_n (int): Number of top categories to display. Default is 10.
    """
    plt.figure(figsize=(10, 6))
    
    # Get value counts and limit to top_n
    value_counts = df[column].value_counts().head(top_n)
    
    sns.countplot(x=value_counts.values, y=value_counts.index, palette='viridis')
    plt.title(f'Distribution of {column}', fontsize=14)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.tight_layout()
    plt.show()


def plot_categorical_distributions(
    df: pd.DataFrame,
    categorical_features: list,
    top_n: int = 10,
    num_cols: int = 2
) -> None:
    """
    Plots the distribution of multiple categorical features.

    Args:
        df: Pandas DataFrame containing the dataset.
        categorical_features: List of categorical column names.
        top_n: Number of top categories to display for each feature.
        num_cols: Number of columns for the subplot grid.
    """
    num_features = len(categorical_features)
    
    if num_features == 0:
        print("No categorical features to plot.")
        return
    
    num_rows = (num_features + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(14, 5 * num_rows))
    
    # Flatten axes for easy iteration and ensure it's always indexable
    axes_flat = axes.flatten() if num_features > 1 else [axes]
    
    for idx, (ax, feature) in enumerate(zip(axes_flat[:num_features], categorical_features)):
        # Get value counts and limit to top_n, then sort descending
        value_counts = df[feature].value_counts().head(top_n).sort_values(ascending=False)
        
        # Calculate percentages
        percentages = (value_counts / len(df) * 100).round(2)
        
        # Create bar plot with a nicer palette and unique color per bar
        colors = sns.color_palette("flare", len(value_counts))
        ax.barh(value_counts.index, value_counts.values, color=colors, edgecolor='black')
        ax.invert_yaxis()
        ax.set_title(f'Distribution of {feature}', fontsize=12)
        ax.set_xlabel('Count', fontsize=10)
        ax.set_ylabel(feature, fontsize=10)
        
        # Add percentage labels
        for j, (count, pct) in enumerate(zip(value_counts.values, percentages.values)):
            ax.text(count + 0.5, j, f'({pct}%)', va='center', fontsize=8)
    
    # Hide empty subplots
    for ax in axes_flat[num_features:]:
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def plot_numerical_distribution(df: pd.DataFrame, column: str, bins: int = 30) -> None:
    """Visualize the distribution of a numeric variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The name of the numeric column to visualize.
        bins (int): Number of bins for the histogram. Default is 30.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    sns.histplot(df[column], bins=bins, kde=True, ax=axes[0], color='skyblue')
    axes[0].set_title(f'Histogram of {column}', fontsize=14)
    axes[0].set_xlabel(column, fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    
    # Boxplot
    sns.boxplot(x=df[column], ax=axes[1], color='lightgreen')
    axes[1].set_title(f'Boxplot of {column}', fontsize=14)
    axes[1].set_xlabel(column, fontsize=12)
    
    plt.tight_layout()
    plt.show()


def compute_percentage_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """_summary_

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    missing_values = pd.DataFrame(df.isna().sum() / df.shape[0])

    missing_values.reset_index(inplace=True)
    missing_values = missing_values.rename(columns = {'index':'feature',
                                                                     0: 'prct_missing'})
 
    missing_values = missing_values.sort_values(by='prct_missing', ascending=False)
    return missing_values


def plot_numerical_distributions(
    data: pd.DataFrame,
    numerical_features: list,
    bins: int = 30,
    num_cols: int = 2
) -> None:
    """
    Plots the distribution of numerical features using histograms and boxplots.

    Args:
        data: Pandas DataFrame containing the dataset.
        numerical_features: List of numerical column names.
        bins: Number of bins for the histogram.
        num_cols: Number of columns for the subplot grid.
    """
    # Filter out features with all missing values
    valid_features = [f for f in numerical_features if not data[f].isnull().all()]
    
    if len(valid_features) == 0:
        print("No valid features to plot.")
        return

    num_features = len(valid_features)
    num_rows = (num_features + num_cols - 1) // num_cols

    fig, axes = plt.subplots(num_rows, num_cols * 2, figsize=(14, 5 * num_rows))
    
    # Handle single row case
    if num_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, feature in enumerate(valid_features):
        row = idx // num_cols
        col = idx % num_cols * 2
        
        # Histogram with KDE curve
        sns.histplot(data[feature], kde=True, bins=bins, ax=axes[row, col], color='skyblue')
        axes[row, col].set_title(f"Histogram of {feature}", fontsize=12)
        axes[row, col].set_xlabel(feature)
        axes[row, col].set_ylabel("Frequency")

        # Boxplot to detect outliers
        sns.boxplot(x=data[feature], ax=axes[row, col + 1], color='lightgreen')
        axes[row, col + 1].set_title(f"Boxplot of {feature}", fontsize=12)

    # Hide empty subplots
    for idx in range(num_features, num_rows * num_cols):
        row = idx // num_cols
        col = idx % num_cols * 2
        axes[row, col].axis('off')
        axes[row, col + 1].axis('off')

    plt.tight_layout()
    plt.show()


def compute_numerical_summary(
    data: pd.DataFrame,
    numerical_features: list,
    cardinality_threshold: int = 10
) -> pd.DataFrame:
    """
    Computes summary statistics for numerical features.

    Args:
        data: Pandas DataFrame containing the dataset.
        numerical_features: List of numerical column names.
        cardinality_analysis_threshold: Threshold to identify potential categorical features.

    Returns:
        pd.DataFrame: Summary statistics for all numerical features.
    """
    summary_stats = []

    for feature in numerical_features:
        # Skip if feature has all missing values
        if data[feature].isnull().all():
            print(f"Skipping '{feature}' - all values are missing.")
            continue

        # Calculate statistics
        missing_count = data[feature].isnull().sum()
        missing_pct = (missing_count / len(data)) * 100
        skewness = data[feature].skew()
        kurtosis = data[feature].kurtosis()
        num_unique = data[feature].nunique()

        # Print detailed statistics
        print(f"\n{'='*50}")
        print(f"Statistics for {feature}:")
        print(f"{'='*50}")
        print(f"Missing Values: {missing_count} ({missing_pct:.2f}%)")
        print(f"Skewness: {skewness:.2f}")
        print(f"Kurtosis: {kurtosis:.2f}")
        print(f"Unique values: {num_unique}")

        # Cardinality analysis
        if num_unique < cardinality_threshold:
            print(f"⚠️  Feature '{feature}' might be categorical (unique < {cardinality_threshold})")

        # Percentiles
        print(f"\nPercentiles:")
        print(f"  5%:  {data[feature].quantile(0.05):.2f}")
        print(f"  25%: {data[feature].quantile(0.25):.2f}")
        print(f"  50%: {data[feature].quantile(0.50):.2f}")
        print(f"  75%: {data[feature].quantile(0.75):.2f}")
        print(f"  95%: {data[feature].quantile(0.95):.2f}")

        # Collect summary stats
        summary_stats.append({
            'feature': feature,
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'unique_values': num_unique,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'mean': data[feature].mean(),
            'std': data[feature].std(),
            'min': data[feature].min(),
            'max': data[feature].max()
        })

    # Return summary DataFrame
    return pd.DataFrame(summary_stats)


def plot_numerical_distribution_by_category(
    df: pd.DataFrame,
    numerical_features: list,
    categorical_column: str,
    plot_type: str = 'boxplot',
    num_cols: int = 2,
    figsize: tuple = None
) -> None:
    """
    Plot the distribution of numerical variables grouped by a categorical variable.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_features (list): List of numerical column names to plot.
        categorical_column (str): The name of the categorical column to group by.
        plot_type (str): Type of plot ('boxplot' or 'violin'). Default is 'boxplot'.
        num_cols (int): Number of columns for the subplot grid. Default is 2.
        figsize (tuple): Figure size (width, height). If None, calculated automatically.
    """
    # Filter out features with all missing values
    valid_features = [f for f in numerical_features if not df[f].isnull().all()]
    
    if len(valid_features) == 0:
        print("No valid numerical features to plot.")
        return
    
    num_features = len(valid_features)
    num_rows = (num_features + num_cols - 1) // num_cols
    
    if figsize is None:
        figsize = (6 * num_cols, 5 * num_rows)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Handle single subplot case
    if num_features == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, feature in enumerate(valid_features):
        ax = axes[idx]
        
        if plot_type == 'boxplot':
            sns.boxplot(data=df, x=categorical_column, y=feature, palette='viridis', ax=ax)
            ax.set_title(f'Boxplot of {feature} by {categorical_column}', fontsize=12)
        elif plot_type == 'violin':
            sns.violinplot(data=df, x=categorical_column, y=feature, palette='viridis', ax=ax)
            ax.set_title(f'Violin Plot of {feature} by {categorical_column}', fontsize=12)
        else:
            raise ValueError("plot_type must be 'boxplot' or 'violin'")
        
        ax.set_xlabel(categorical_column, fontsize=10)
        ax.set_ylabel(feature, fontsize=10)
        ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for idx in range(num_features, num_rows * num_cols):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()


def compute_categorical_summary(
    data: pd.DataFrame,
    categorical_features: list,
    cardinality_threshold: int = 10
) -> pd.DataFrame:
    """
    Computes summary statistics for categorical features.

    Args:
        data: Pandas DataFrame containing the dataset.
        categorical_features: List of categorical column names.
        cardinality_threshold: Threshold to identify high-cardinality features.

    Returns:
        pd.DataFrame: Summary statistics for all categorical features.
    """
    summary_stats = []

    for feature in categorical_features:
        # Skip if feature has all missing values
        if data[feature].isnull().all():
            print(f"Skipping '{feature}' - all values are missing.")
            continue

        # Calculate statistics
        missing_count = data[feature].isnull().sum()
        missing_pct = (missing_count / len(data)) * 100
        num_unique = data[feature].nunique()
        mode_value = data[feature].mode()[0] if not data[feature].mode().empty else None
        mode_count = data[feature].value_counts().iloc[0] if not data[feature].empty else 0

        # Print detailed statistics
        print(f"\n{'='*50}")
        print(f"Statistics for {feature}:")
        print(f"{'='*50}")
        print(f"Missing Values: {missing_count} ({missing_pct:.2f}%)")
        print(f"Unique values: {num_unique}")
        print(f"Mode: {mode_value} (count: {mode_count})")

        # Cardinality analysis
        if num_unique > cardinality_threshold:
            print(f"⚠️  High cardinality (unique > {cardinality_threshold})")
        
        if num_unique == 2:
            print(f"ℹ️  Binary feature detected")
        
        if num_unique == 1:
            print(f"⚠️  Constant feature (only one unique value)")

        # Value counts
        print(f"\nTop 10 value counts:")
        value_counts = data[feature].value_counts().head(10)
        for value, count in value_counts.items():
            pct = (count / len(data)) * 100
            print(f"  {value}: {count} ({pct:.2f}%)")

        # Collect summary stats
        summary_stats.append({
            'feature': feature,
            'missing_count': missing_count,
            'missing_pct': missing_pct,
            'unique_values': num_unique,
            'mode': mode_value,
            'mode_count': mode_count,
            'mode_pct': (mode_count / len(data)) * 100
        })

    # Return summary DataFrame
    return pd.DataFrame(summary_stats)


def plot_correlation_analysis(
    data: pd.DataFrame,
    numerical_features: list,
    method: str = 'pearson',
    threshold: float = 0.7,
    figsize: tuple = (12, 10)
) -> pd.DataFrame:
    """
    Plots the correlation analysis of numerical features.

    Args:
        data: Pandas DataFrame containing the dataset.
        numerical_features: List of numerical column names.
        method: Correlation method ('pearson', 'spearman', or 'kendall').
        threshold: Threshold for highlighting highly correlated features.
        figsize: Figure size for the heatmap.

    Returns:
        pd.DataFrame: Correlation matrix.
    """
    # Compute correlation matrix
    data_subset = data[numerical_features]
    corr_matrix = data_subset.corr(method=method)
    
    # Plot heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        square=True,
        linewidths=0.5,
        cbar_kws={'shrink': 0.8}
    )
    plt.title(f'Correlation Matrix ({method.capitalize()})', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Find highly correlated pairs
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            value = corr_matrix.iloc[i, j]
            if pd.notnull(value) and abs(float(value)) >= threshold:
                high_corr_pairs.append({
                    'feature_1': corr_matrix.columns[i],
                    'feature_2': corr_matrix.columns[j],
                    'correlation': value
                })
    
    # Print highly correlated pairs
    if high_corr_pairs:
        print(f"\n⚠️  Highly correlated pairs (|r| >= {threshold}):")
        print("=" * 60)
        for pair in high_corr_pairs:
            print(f"  {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.3f}")
    else:
        print(f"\n✓ No highly correlated pairs found (threshold: {threshold})")
    
    return corr_matrix


def plot_categorical_distribution_by_category(
    df: pd.DataFrame,
    categorical_features: list,
    groupby_column: list,
    num_cols: int = 2,
    figsize: tuple = None,
    plot_type: str = 'count'
) -> None:
    """
    Plot the distribution of categorical features grouped by one or more categorical variables.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_features (list): List of categorical column names to plot.
        groupby_column (list): List of categorical column names to group by. If multiple columns provided, they are combined into a multi-index grouping.
        num_cols (int): Number of columns for the subplot grid. Default is 2.
        figsize (tuple): Figure size (width, height). If None, calculated automatically.
        plot_type (str): Type of plot ('count' or 'percentage'). Default is 'count'.
    """
    # Ensure groupby_column is a list
    if isinstance(groupby_column, str):
        groupby_column = [groupby_column]
    
    # Filter out features with all missing values
    valid_features = [f for f in categorical_features if not df[f].isnull().all()]
    
    if len(valid_features) == 0:
        print("No valid categorical features to plot.")
        return
    
    num_features = len(valid_features)
    num_rows = (num_features + num_cols - 1) // num_cols
    
    if figsize is None:
        figsize = (6 * num_cols, 5 * num_rows)
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    
    # Normalize axes to a flat list regardless of subplot shape
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    
    groupby_label = ', '.join(groupby_column)
    groupby_data = df[groupby_column] if len(groupby_column) > 1 else df[groupby_column[0]]
    
    for idx, feature in enumerate(valid_features):
        ax = axes[idx]
        
        # Create contingency table, grouped by one or multiple columns
        contingency = pd.crosstab(df[feature], groupby_data)
        
        if plot_type == 'count':
            contingency.plot(kind='bar', ax=ax, width=0.8)
            ax.set_title(f'{feature} by {groupby_label} (Count)', fontsize=12)
            ax.set_ylabel('Count', fontsize=10)
        elif plot_type == 'percentage':
            # Calculate percentage within each category
            contingency_pct = contingency.div(contingency.sum(axis=1), axis=0) * 100
            contingency_pct.plot(kind='bar', ax=ax, stacked=True, width=0.8)
            ax.set_title(f'{feature} by {groupby_label} (Percentage)', fontsize=12)
            ax.set_ylabel('Percentage', fontsize=10)
        else:
            raise ValueError("plot_type must be 'count' or 'percentage'")
        
        ax.set_xlabel(feature, fontsize=10)
        ax.legend(title=groupby_label, fontsize=8, title_fontsize=9, loc='best')
        ax.tick_params(axis='x', rotation=45)
    
    # Hide empty subplots
    for idx in range(num_features, num_rows * num_cols):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.show()
