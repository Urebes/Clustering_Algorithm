import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#load the csv file
#return X (without labels)
def get_dataset(path):
    df = pd.read_csv(path,header = 0)

    #drop the label column
    df.drop(['ca_cervix'], axis=1, inplace=True)
    cols = df.columns
    ms = MinMaxScaler()
    df = ms.fit_transform(df)
    df = pd.DataFrame(df, columns=[cols])
    return df

#load the csv file and return only the labels
def get_labels(path):
    df = pd.read_csv(path,header = 0)
    return df['ca_cervix']





