import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Optional
from backend.utils.read_csv import _read_csv_with_fallback

def generate_histogram(file_contents: bytes, column_name: str):
    try:
        df = _read_csv_with_fallback(file_contents)
        if column_name not in df.columns:
            return "column_not_found"
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            return "column_not_numeric"

        plt.figure(figsize=(10, 6))
        sns.histplot(df[column_name], kde=True)
        plt.title(f'Histogram of {column_name}', fontsize=16)
        plt.xlabel(column_name, fontsize=12)
        plt.ylabel('Frequency', fontsize=12)

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)

        return img_buffer.getvalue()

    except Exception as e:
        print(f"Error generating histogram: {e}")
        return None

def generate_correlation_heatmap(file_contents: bytes):
    try:
        df = _read_csv_with_fallback(file_contents)
        df_numeric = df.select_dtypes(include=['number'])
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_numeric.corr(), annot=True, cmap='coolwarm', fmt=".2f")
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        return img_buffer.getvalue()
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return None

def generate_missing_value_heatmap(file_contents: bytes):
    try:
        df = _read_csv_with_fallback(file_contents)

        plt.figure(figsize=(12, 8))
        sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
        plt.title('Heatmap of Missing Values', fontsize=16)
        plt.xlabel('Columns', fontsize=12)

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)

        return img_buffer.getvalue()

    except Exception as e:
        print(f"Error generating missing value heatmap: {e}")
        return None

def generate_target_feature_plot(file_contents: bytes, target_column: str, feature_column: str):
    try:
        df = _read_csv_with_fallback(file_contents)
        if df is None: return None

        if target_column not in df.columns or feature_column not in df.columns:
            return "column_not_found"

        plt.figure(figsize=(12, 8))

        if pd.api.types.is_numeric_dtype(df[feature_column]):
            sns.boxplot(data=df, x=target_column, y=feature_column)
            plt.title(f'Distribusi {feature_column} berdasarkan {target_column}', fontsize=16)
        else:
            sns.countplot(data=df, x=feature_column, hue=target_column)
            plt.title(f'Jumlah {feature_column} berdasarkan {target_column}', fontsize=16)
            plt.xticks(rotation=45, ha='right')

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)

        return img_buffer.getvalue()
    except Exception as e:
        print(f"Error generating target-feature plot: {e}")
        return None