import pandas as pd


data_path = 'data.csv'
data = pd.read_csv(data_path)       # dataframe

print(data)


# ------ inspect data -------
    # features
    # data points

print(data.shape)           # tuple (rows, columns)
                            # (10,4) - 3 features, 30 data points

# range of features
    # max - min of each feature

features_columns = [col for col in data if col != 'y']
features_range = {}
for feature in features_columns:
    min_value = data[feature].min()
    max_value = data[feature].max()
    features_range[feature] = float(max_value - min_value)

print("range of features: ")
for feature in features_range.items():
    print(feature)