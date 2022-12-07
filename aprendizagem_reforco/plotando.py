import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df_ia = pd.read_csv('output_test_data.csv')
#df_rand = pd.read_csv ('output_test_data01.csv')
df_ia["Parameter_RL"] = df_ia["Parameter_RL"].apply(lambda x: (x/100))
df_ia["Reward"]= df_ia["Reward"].apply(lambda x: (x*100))
df_ia["Eficiency"]= df_ia["Eficiency"].apply(lambda x: (x*100))


df_ia.plot()

plt.title("Otimização pela Impedância da carga ")

plt.savefig("results.png")

