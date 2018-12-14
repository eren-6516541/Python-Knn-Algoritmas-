import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.neighbors import KNeighborsClassifier

veri=pd.read_csv("c:\\dataset.csv", header=0)

egitim=veri.sample(frac=0.8,replace=False)
test=veri.drop(egitim.index)

   
knn = KNeighborsClassifier()

girdiler = ['Yas', 'TahminiMaas']

knn.fit(egitim[girdiler], egitim['SatinAldiMi'])

tahmin = knn.predict(test[girdiler])
tahmin==test['SatinAldiMi']   

print("Test sonucu: {:.2f}".format(np.mean(tahmin == test['SatinAldiMi'])))

sns.FacetGrid(veri, hue="SatinAldiMi", size=4).map(plt.scatter, "Yas", "TahminiMaas").add_legend()

 
 