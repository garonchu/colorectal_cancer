import yaml
from process.preprocessing import load_data
from process.preprocessing import one_hot
from process.preprocessing import split_X_y
from process.training import split_train_test

# Load YAML config
with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)

data_path = config["data"]["data_path"]
y_column_name = "Survival_Status"
test_size = 0.2
random_state = 123

def main():

    # Step 1: Load dataset
    df = load_data(data_path)
    df = df.drop(columns=["Patient_ID"]) #temp hack

    # Step 2: split into X and y df
    X, y = split_X_y(df, y_column_name)

    # Step 3: one hot encode
    encoded_X = one_hot(X)

    # Step 4: split into train_test sets
    X_train, X_test, y_train, y_test = split_train_test(encoded_X
                                                        ,y
                                                        ,test_size = test_size
                                                        ,random_state=random_state
                                                        ,stratify=y)
    
    print(encoded_X.info())


if __name__ == "__main__":
    main()