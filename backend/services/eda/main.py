import pandas as pd
import io
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.outliers_influence import variance_inflation_factor
from typing import Optional


def _read_csv_with_fallback(file_contents: bytes):
    """Fungsi helper untuk membaca CSV dengan fallback encoding."""
    try:
        return pd.read_csv(io.StringIO(file_contents.decode('utf-8')))
    except UnicodeDecodeError:
        print("UTF-8 decoding failed, falling back to latin-1.")
        return pd.read_csv(io.StringIO(file_contents.decode('latin-1')))
    except Exception as e:
        print(f"Failed to read CSV with all encodings: {e}")
        return None

def get_csv_description(file_contents: bytes):
    try:
        df = _read_csv_with_fallback(file_contents)
        description = df.describe().to_dict()
        return description
    except Exception as e:
        print(f"Error processing file: {e}")
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

def generate_histogram(file_contents: bytes, column_name: str):
    """
    Menerima konten file CSV dan nama kolom, membuat gambar histogram,
    dan mengembalikannya dalam bentuk bytes.
    """
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

def generate_missing_value_heatmap(file_contents: bytes):
    """
    Menerima konten file CSV, membuat heatmap dari missing values,
    dan mengembalikannya dalam bentuk bytes. (Fitur #11)
    """
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

def get_outliers(file_contents: bytes, column_name: str):
    """
    Mengidentifikasi outliers pada kolom numerik menggunakan metode IQR
    dan mengembalikannya dalam bentuk dictionary. (Bagian dari Fitur #2)
    """
    try:
        df = _read_csv_with_fallback(file_contents)

        if column_name not in df.columns:
            return "column_not_found"
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            return "column_not_numeric"

        Q1 = df[column_name].quantile(0.25)
        Q3 = df[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = df[(df[column_name] < lower_bound) | (df[column_name] > upper_bound)]
        
        return {
            "column_name": column_name,
            "outlier_count": len(outliers),
            "outliers_list": outliers[column_name].tolist()
        }

    except Exception as e:
        print(f"Error getting outliers: {e}")
        return None

def get_skewness(file_contents: bytes):
    """
    Menghitung skewness untuk semua kolom numerik dan
    mengembalikannya dalam format dictionary. (Bagian dari Fitur #2)
    """
    try:
        df = _read_csv_with_fallback(file_contents)
        df_numeric = df.select_dtypes(include=['number'])
        
        skewness = df_numeric.skew().to_dict()
        return skewness

    except Exception as e:
        print(f"Error calculating skewness: {e}")
        return None

def get_categorical_insights(file_contents: bytes, column_name: str):
    """
    Menganalisis kolom kategorikal untuk mendapatkan jumlah nilai unik
    dan frekuensi setiap kategori. (Fitur #9)
    """
    try:
        df = _read_csv_with_fallback(file_contents)
        if df is None: return None

        if column_name not in df.columns:
            return "column_not_found"
        
        # Kolom numerik pun bisa dianggap kategorikal jika jumlah uniknya sedikit
        if pd.api.types.is_numeric_dtype(df[column_name]) and df[column_name].nunique() > 50:
             return "column_is_highly_numeric"

        unique_count = df[column_name].nunique()
        value_counts = df[column_name].value_counts().to_dict()

        return {
            "column_name": column_name,
            "unique_values_count": unique_count,
            "value_counts": value_counts
        }
    except Exception as e:
        print(f"Error getting categorical insights: {e}")
        return None

def analyze_target(file_contents: bytes, target_column: str):
    """
    Menganalisis variabel target untuk menentukan tipe masalah (Klasifikasi/Regresi)
    dan memeriksa ketidakseimbangan kelas. (Fitur #10 & #3)
    """
    try:
        df = _read_csv_with_fallback(file_contents)
        if df is None: return None

        if target_column not in df.columns:
            return "column_not_found"

        target_series = df[target_column].dropna() # Hapus missing values untuk analisis
        problem_type = "Klasifikasi"
        
        # Logika penentuan tipe masalah
        if pd.api.types.is_numeric_dtype(target_series) and target_series.nunique() > 15:
            problem_type = "Regresi"

        analysis = {"problem_type": problem_type}

        if problem_type == "Klasifikasi":
            distribution = target_series.value_counts(normalize=True).to_dict()
            analysis["class_distribution"] = distribution
            # Cek imbalance jika ada kelas dengan porsi < 20%
            is_imbalanced = any(v < 0.2 for v in distribution.values())
            analysis["is_imbalanced"] = is_imbalanced
        else: # Regresi
            analysis["distribution_summary"] = target_series.describe().to_dict()

        return analysis
    except Exception as e:
        print(f"Error analyzing target variable: {e}")
        return None

def generate_target_feature_plot(file_contents: bytes, target_column: str, feature_column: str):
    """
    Membuat visualisasi hubungan antara fitur dan target. (Fitur #4)
    - Boxplot untuk fitur numerik.
    - Countplot untuk fitur kategorikal.
    """
    try:
        df = _read_csv_with_fallback(file_contents)
        if df is None: return None

        if target_column not in df.columns or feature_column not in df.columns:
            return "column_not_found"

        plt.figure(figsize=(12, 8))
        
        # Jika fitur adalah numerik, gunakan boxplot
        if pd.api.types.is_numeric_dtype(df[feature_column]):
            sns.boxplot(data=df, x=target_column, y=feature_column)
            plt.title(f'Distribusi {feature_column} berdasarkan {target_column}', fontsize=16)
        # Jika fitur adalah kategorikal, gunakan countplot
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

def run_full_data_profile(file_contents: bytes):
    """
    Menjalankan serangkaian analisis EDA komprehensif secara otomatis
    dan mengembalikan hasilnya dalam satu JSON besar. (Fitur #8)
    """
    try:
        df = _read_csv_with_fallback(file_contents)
        if df is None: return None

        # 1. Ringkasan Statistik
        description = df.describe().to_dict()

        # 2. Info Tipe Data dan Nilai Non-Null
        info_buffer = io.StringIO()
        df.info(buf=info_buffer)
        info_str = info_buffer.getvalue()

        # 3. Ringkasan Missing Values
        missing_values = df.isnull().sum().to_dict()
        missing_percentage = (df.isnull().sum() / len(df) * 100).to_dict()

        # 4. Skewness
        skewness = get_skewness(file_contents)
        
        return {
            "data_shape": {"rows": df.shape[0], "columns": df.shape[1]},
            "statistical_summary": description,
            "data_info": info_str,
            "missing_values_summary": {
                "count": missing_values,
                "percentage": missing_percentage
            },
            "skewness": skewness
        }
    except Exception as e:
        print(f"Error running full data profile: {e}")
        return None

def calculate_vif(file_contents: bytes):
    """
    Menghitung Variance Inflation Factor (VIF) untuk mendeteksi multikolinearitas.
    """
    try:
        df = _read_csv_with_fallback(file_contents)
        if df is None: return None

        # Hanya gunakan kolom numerik dan hapus baris dengan missing values
        df_numeric = df.select_dtypes(include=['number']).dropna()

        # VIF membutuhkan minimal 2 kolom
        if df_numeric.shape[1] < 2:
            return {"message": "VIF membutuhkan setidaknya 2 kolom numerik."}

        # Hitung VIF
        vif_data = pd.DataFrame()
        vif_data["feature"] = df_numeric.columns
        vif_data["VIF"] = [variance_inflation_factor(df_numeric.values, i) for i in range(df_numeric.shape[1])]

        return vif_data.to_dict('records')
    except Exception as e:
        print(f"Error calculating VIF: {e}")
        return None

def generate_custom_plot(
    file_contents: bytes,
    plot_type: str,
    x_col: str,
    y_col: Optional[str] = None,
    hue_col: Optional[str] = None,
    orientation: str = 'v'
):
    """
    Satu fungsi yang lebih andal untuk membuat berbagai jenis plot yang bisa dikustomisasi.
    """
    try:
        df = _read_csv_with_fallback(file_contents)
        if df is None:
            return "error: Gagal membaca file CSV."

        # 1. Validasi Keberadaan Kolom
        required_cols = [c for c in [x_col, y_col, hue_col] if c is not None]
        for col in required_cols:
            if col not in df.columns:
                return f"error: Kolom '{col}' tidak ditemukan di dalam dataset."

        # 2. Validasi Kompatibilitas Tipe Plot dan Data
        is_x_numeric = pd.api.types.is_numeric_dtype(df[x_col])
        is_y_numeric = y_col and pd.api.types.is_numeric_dtype(df[y_col])

        if plot_type in ['scatter', 'box'] and not y_col:
            return f"error: Plot tipe '{plot_type}' membutuhkan kolom sumbu Y (y_col)."
        if plot_type == 'scatter' and not (is_x_numeric and is_y_numeric):
            return "error: Scatter plot membutuhkan kolom X dan Y yang keduanya numerik."
        if plot_type == 'box' and not is_y_numeric:
             return "error: Box plot membutuhkan kolom Y yang numerik."

        # 3. Proses Plotting yang Disempurnakan
        plt.figure(figsize=(12, 8))
        plot_kwargs = {'data': df, 'x': x_col, 'y': y_col, 'hue': hue_col}

        # Penanganan orientasi horizontal yang lebih cerdas
        if orientation == 'h' and plot_type in ['bar', 'box']:
            plot_kwargs['x'], plot_kwargs['y'] = y_col, x_col
            plot_kwargs['orient'] = 'h'

        # Pemilihan plot
        if plot_type == 'histogram':
            sns.histplot(data=df, x=x_col, hue=hue_col, kde=True)
        elif plot_type == 'bar':
            # Jika tidak ada sumbu Y, gunakan countplot untuk menghitung frekuensi
            if y_col is None:
                 sns.countplot(**{k: v for k, v in plot_kwargs.items() if k != 'y'})
            else: # Jika sumbu Y diberikan, gunakan barplot standar (agregasi rata-rata)
                 sns.barplot(**plot_kwargs)
        elif plot_type == 'box':
            sns.boxplot(**plot_kwargs)
        elif plot_type == 'scatter':
            sns.scatterplot(**plot_kwargs)
        else:
            return f"error: Tipe plot '{plot_type}' tidak didukung."

        # 4. Peningkatan Estetika Plot
        title = f'{plot_type.capitalize()} of {x_col}'
        if y_col: title += f' by {y_col}'
        if hue_col: title += f' (colored by {hue_col})'

        plt.title(title, fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout() # Mencegah label tumpang tindih

        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format='png', bbox_inches='tight')
        plt.close()
        img_buffer.seek(0)
        return img_buffer.getvalue()

    except Exception as e:
        # 5. Penanganan Error yang Lebih Spesifik
        print(f"Error saat generate_custom_plot: {e}")
        return f"error: Terjadi kesalahan saat membuat plot - {str(e)}"