from src.train import __get_data

X_train, X_test, y_train, y_test = __get_data()
X_train.to_csv("data/processed/X_train.csv", index=False)
y_train.to_csv("data/processed/y_train.csv", index=False)
X_test.to_csv("data/processed/X_test.csv", index=False)
y_test.to_csv("data/processed/y_test.csv", index=False)
