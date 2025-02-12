import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def load_data(file_path):
    return pd.read_csv(file_path)


def one_hot(df):
    """One hot encode categorical columns"""
    
    # Drop missing values (or handle them differently if needed)
    #df = df.dropna()
    
    # One-hot encode categorical features
    cat_columns = df.select_dtypes(include=["object"]).columns  # Detect categorical columns
    encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    cat_encoded = encoder.fit_transform(df[cat_columns])

    # Convert to DataFrame and merge with numeric features
    encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out(cat_columns))
    df_numeric = df.select_dtypes(exclude=["object"])  # Keep numeric columns
    df_final = pd.concat([df_numeric, encoded_df], axis=1)  # Final dataset
    return df_final

def split_X_y(df, yCol):
    X = df.drop(columns=[yCol])
    y = df[yCol]
    return X, y