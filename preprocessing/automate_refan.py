import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

def preprocessing_data(dataset_raw_path):

    # Loading dataset
    mushroom_df = pd.read_csv(dataset_raw_path)

    # Melakukan proses encoding data
    label_encoder = LabelEncoder()
    categorical_features = mushroom_df.columns
    mushroom_encoder_df = pd.DataFrame(mushroom_df)
    for col in categorical_features:
        mushroom_encoder_df[col] = label_encoder.fit_transform(mushroom_encoder_df[col])

    # Melakukan prose standardisasi
    scaler = MinMaxScaler()
    scaler_features_without_class = mushroom_df.drop(columns='class').columns
    mushroom_scaler_df = pd.DataFrame(mushroom_encoder_df)
    mushroom_scaler_df[scaler_features_without_class] = scaler.fit_transform(
        mushroom_scaler_df[scaler_features_without_class])

    return mushroom_scaler_df

if __name__ == "__main__":
    raw_dataset_path = "mushrooms_raw.csv"
    processed_df = preprocessing_data(raw_dataset_path)

    print("Preprocessing Complete")

    # Menyimpan preprocessed dataset dengan format csv
    processed_df.to_csv("mushrooms_preprocessed.csv", index=False)