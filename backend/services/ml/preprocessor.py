import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def preprocess_data(df: pd.DataFrame, target_column_name: str):
    """
    Membersihkan dan mempersiapkan data untuk pelatihan model.

    Args:
        df: DataFrame pandas yang akan diproses.
        target_column_name: Nama kolom target.

    Returns:
        Tuple berisi: X_processed, y, preprocessor_pipeline
    """
    # 1. Pisahkan fitur (X) dan target (y)
    X = df.drop(columns=[target_column_name])
    y = df[target_column_name]

    # 2. Identifikasi tipe kolom pada fitur
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # 3. Buat pipeline preprocessing untuk setiap tipe data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # 4. Gabungkan kedua pipeline menjadi satu preprocessor
    preprocessor_pipeline = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # 5. Terapkan preprocessor pada data fitur
    X_processed = preprocessor_pipeline.fit_transform(X)
    
    return X_processed, y, preprocessor_pipeline