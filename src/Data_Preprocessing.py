from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt
from math import sin, cos, sqrt, atan2, radians


def load_data(path):
    path = Path(path)  # Ensures compatibility on all OS
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path.resolve()}")
    return pd.read_csv(path, keep_default_na=False, na_values=["", "NA", "N/A"])


def process_arrest_data(df, min_year, top_n_offenses):
    """
    Filters, transforms, and pivots NYPD arrest data.

    Parameters:
    - df (pd.DataFrame): The input DataFrame containing arrest records.
    - min_year (int): Minimum year to include in the dataset.
    - top_n_offenses (int): Number of most frequent offenses to keep.

    Returns:
    - pd.DataFrame: Transformed wide-format DataFrame with top offenses as columns.
    """

    # Step 1: Filter to relevant columns
    relevant_cols = ["ARREST_DATE", "OFNS_DESC", "ARREST_BORO", "ARREST_PRECINCT", "JURISDICTION_CODE"]
    df = df[relevant_cols].copy()

    # Step 2: Parse ARREST_DATE and filter to min_year
    df['ARREST_DATE'] = pd.to_datetime(df['ARREST_DATE'], errors='coerce')
    df = df[df['ARREST_DATE'] >= pd.Timestamp(f'{min_year}-01-01')]
    df['year'] = df['ARREST_DATE'].dt.year
    df['month'] = df['ARREST_DATE'].dt.month
    df['year_month'] = df['ARREST_DATE'].dt.to_period('M').astype(str)
    df = df.drop(columns=["ARREST_DATE"])

    # Step 3: Keep only top N offenses
    top_offenses = df['OFNS_DESC'].value_counts().nlargest(top_n_offenses).index
    df = df[df['OFNS_DESC'].isin(top_offenses)]

    # Step 4: Group and pivot to wide format
    df_grouped = (
        df.groupby(["ARREST_BORO", "ARREST_PRECINCT", "JURISDICTION_CODE", "year", "month", "year_month", "OFNS_DESC"])
        .size()
        .reset_index(name="count")
    )

    df_pivot = df_grouped.pivot_table(
        index=["ARREST_BORO", "ARREST_PRECINCT", "JURISDICTION_CODE", "year", "month", "year_month"],
        columns="OFNS_DESC",
        values="count",
        fill_value=0
    ).reset_index()

    # Step 5: Flatten column names
    df_pivot.columns.name = None

    return df_pivot



def perform_pca_analysis(df, exclude_cols=None, low_variance_threshold=0.01, top_n=3, plot_corr=True, plot_variance=True):
    """
    Perform PCA analysis on a DataFrame, excluding specified metadata columns.

    Parameters:
    - df (pd.DataFrame): The input DataFrame (wide format with numeric columns).
    - exclude_cols (list): List of columns to exclude (metadata like IDs, dates).
    - low_variance_threshold (float): Threshold below which variance is considered too low (optional removal).
    - top_n (int): Number of top variables to report for each principal component.
    - plot_corr (bool): Whether to display a correlation heatmap.
    - plot_variance (bool): Whether to plot the explained variance.

    Returns:
    - loadings (pd.DataFrame): Component loadings for each feature.
    - important_vars (dict): Dictionary of top contributing features for each PC.
    """
    if exclude_cols is None:
        exclude_cols = []

    # 1. Select numeric columns only
    numeric_df = df[[col for col in df.columns if col not in exclude_cols]].copy()

    # 2. Remove low-variance columns
    low_variance_cols = numeric_df.var()[numeric_df.var() < low_variance_threshold].index.tolist()
    print("Columns with low variance (optional to drop):", low_variance_cols)
    numeric_df.drop(columns=low_variance_cols, inplace=True)

    # 3. Correlation matrix (optional)
    if plot_corr:
        corr_matrix = numeric_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=False, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.show()

    # 4. Standardize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_df)

    # 5. PCA fitting
    pca = PCA()
    pca.fit(scaled_data)

    # 6. Explained variance
    explained_variance = pca.explained_variance_ratio_
    if plot_variance:
        plt.figure(figsize=(8, 6))
        plt.plot(np.cumsum(explained_variance), marker='o')
        plt.xlabel('Number of Principal Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance by PCA Components')
        plt.grid(True)
        plt.show()

    # 7. Loadings (component weights per feature)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(numeric_df.columns))],
        index=numeric_df.columns
    )

    # 8. Top contributors per PC
    important_vars = {}
    for i in range(1, min(6, len(loadings.columns)+1)):  # Only up to available PCs
        pc = f'PC{i}'
        top_vars = loadings[pc].abs().sort_values(ascending=False).head(top_n).index.tolist()
        important_vars[pc] = top_vars

        print(f"\nTop variable contributions to {pc}:")
        print(loadings[pc].sort_values(ascending=False))

    # 9. Summary
    print("\nRecommended variables based on PCA analysis:")
    for pc, vars in important_vars.items():
        print(f"{pc}: {vars}")

    return loadings, important_vars

def distance(lat1, lon1, lat2, lon2):

    # approximate radius of earth in km
    R = 6373.0

    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    
    return distance


def write_data(df, filename, file_format='csv'):
    """
    Save a DataFrame to the data/processed/ folder.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        filename (str): Output file name (e.g., 'cleaned_data.csv').
        file_format (str): Format to save ('csv' or 'xlsx').

    Returns:
        str: Full path to the saved file.
    """
    processed_path = Path('data/processed')
    processed_path.mkdir(parents=True, exist_ok=True)  # Ensure folder exists

    full_path = processed_path / filename

    if file_format == 'csv':
        df.to_csv(full_path, index=False)
    elif file_format == 'xlsx':
        df.to_excel(full_path, index=False)
    else:
        raise ValueError("Unsupported file format. Use 'csv' or 'xlsx'.")

    print(f"Data written to: {full_path.resolve()}")
    return str(full_path)


