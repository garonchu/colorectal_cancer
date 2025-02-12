import pandas as pd
from sklearn.model_selection import train_test_split

def split_train_test(X, y, test_size, random_state,stratify):
    X_train, X_test, y_train, y_test = train_test_split(X
                                                        ,y
                                                        ,test_size=test_size
                                                        ,random_state=random_state
                                                        ,stratify=stratify
                                                        )
    return X_train, X_test, y_train, y_test