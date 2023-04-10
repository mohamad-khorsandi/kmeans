import random

import pandas as pd
from sample import Sample
from kmeans import Kmeans

if __name__ == "__main__":
    df = pd.read_csv('Mall_Customers.csv', index_col=0)

    df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
    df = (df - df.mean()) / df.std()

    data = []
    for smp_id, ser in df.iterrows():
        tmp_smp = Sample(ser.to_numpy(), smp_id=smp_id)
        data.append(tmp_smp)
    random.shuffle(data)

    km =Kmeans(5, data, 15)

    km.train()
    km.show_nearest(5)
    km.gp_sz_hist()
