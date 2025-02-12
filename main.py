from process.preprocessing import load_data
from process.preprocessing import one_hot


data_path = "colorectal_cancer_prediction.csv"

def main():
    # Step 1: Load dataset
    df = load_data(data_path)

    # Step 2: one hot encode
    encoded_df = one_hot(df)

    print(encoded_df.head())
    



if __name__ == "__main__":
    main()