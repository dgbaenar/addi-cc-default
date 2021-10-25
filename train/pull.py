import pandas as pd


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"

df = pd.read_excel(url, header=1)

df.to_csv("./data/raw.csv", index=False)

print("Data file successfully downloaded.")