import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("/home/borg/Documents/Mestrado/simulador/ML-Retifier/forca_bruta/output_csv/combined.csv")

df.dropna(inplace=True)

for col in df:
	X = col.apply(lambda x: (x.split()[0].replace('+0j)', '')))
	X = X.apply(lambda x: (x.split()[0].replace('(', '')))
	X = X.apply(lambda x: (x.split()[0].replace('0j', '0'))).astype(float)


print(X)


#df = pd.DataFrame.from_dict(target, orient='index').T.to_csv('filtrado.csv', index=True)



