import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#load the csv file
def get_dataset(path):
    df = pd.read_csv(path,header = 0)
    df.drop(['ca_cervix'], axis=1, inplace=True)
    cols = df.columns
    ms = MinMaxScaler()
    df = ms.fit_transform(df)
    df = pd.DataFrame(df, columns=[cols])
    return df

def get_labels(path):
    df = pd.read_csv(path,header = 0)
    return df['ca_cervix']





