#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 18:53:08 2023

@author: claire
"""

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns
import io
from math import prod

import matplotlib.pyplot as plt 
%matplotlib inline
import plotly.express as px
import plotly.graph_objects as go
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


from sklearn.preprocessing import StandardScaler,normalize
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import MeanShift
from sklearn.cluster import estimate_bandwidth
from sklearn import metrics
from sklearn import cluster 
from sklearn.cluster import SpectralClustering

from sklearn.decomposition import PCA

import warnings
warnings.filterwarnings('ignore')



df_cc = pd.read_csv('/Users/claire/Documents/Documents/STUDIES/6. M2 TIDE/Semestre 9/Analyse de données en grande dimension/Projet ADGD/CC GENERAL.csv')             


df_cc.columns

def df_analyse(df, columns, name_df):
    """
    Initial analysis on the DataFrame.

    Parameters
    ----------
    Args:
        df (pandas.DataFrame): DataFrame to analyze.
        columns (list): Dataframe keys in list format.
        name_df (str): DataFrame name.

    Returns:
        None. 
        Print the initial analysis on the DataFrame. 
    """
    
    # Calculating the memory usage based on dataframe.info()
    buf = io.StringIO()
    df.info(buf=buf)
    memory_usage = buf.getvalue().split('\n')[-2]
  
    if df.empty:
        print("The", name_df, "dataset is empty. Please verify the file.")
    else:
        empty_cols = [col for col in df.columns if df[col].isna().all()] # identifying empty columns
        df_rows_duplicates = df[df.duplicated()] #identifying full duplicates rows
        
        # Creating a dataset based on Type object and records by columns
        type_cols = df.dtypes.apply(lambda x: x.name).to_dict() 
        df_resume = pd.DataFrame(list(type_cols.items()), columns = ["Name", "Type"])
        df_resume["Records"] = list(df.count())
        
        print("\nInitial Analysis of", name_df, "dataset")
        print("--------------------------------------------------------------------------")
        print("- Dataset shape:                 ", df.shape[0], "rows and", df.shape[1], "columns")
        print("- Total of NaN values:           ", df.isna().sum().sum())
        print("- Percentage of NaN:             ", round((df.isna().sum().sum() / prod(df.shape)) * 100, 2), "%")
        print("- Total of full duplicates rows: ", df_rows_duplicates.shape[0])
        print("- Total of empty rows:           ", df.shape[0] - df.dropna(axis="rows", how="all").shape[0]) if df.dropna(axis="rows", how="all").shape[0] < df.shape[0] else \
                    print("- Total of empty rows:            0")
        print("- Total of empty columns:        ", len(empty_cols))
        print("  + The empty column is:         ", empty_cols) if len(empty_cols) == 1 else \
                    print("  + The empty column are:         ", empty_cols) if len(empty_cols) >= 1 else None
        
        print("\n- The key(s):", columns, "is not present multiple times in the dataframe.\n  It CAN be used as a primary key.") if df.size == df.drop_duplicates(columns).size else \
                    print("\n- The key(s):", columns, "is present multiple times in the dataframe.\n  It CANNOT be used as a primary key.")
        
        print("\n- Type object and records by columns         (",memory_usage,")")
        print("--------------------------------------------------------------------------")
        print(df_resume.sort_values("Records", ascending=False))
        
        
# Analyse df
df_analyse(df_cc, ["CUST_ID"], "train")

# drop cust_id
df_cc.drop(['CUST_ID'], axis=1, inplace=True)

# Check for NaN
df_cc.isnull().sum().sort_values(ascending=False).head()

"""*********************************************************************************"""
# Imputation statistique => remplace NaN par median
df_cc['MINIMUM_PAYMENTS'] = df_cc['MINIMUM_PAYMENTS'].fillna(df_cc['MINIMUM_PAYMENTS'].median()) 

df_cc['CREDIT_LIMIT'] = df_cc['CREDIT_LIMIT'].fillna(df_cc['CREDIT_LIMIT'].median()) 


"""*********************************************************************************"""
# Imputation KNNImputer

import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, calinski_harabasz_score

df1=df_cc.copy()

df1.fillna(df1.mean(), inplace=True)
df1.isnull().sum().sort_values(ascending=False).head()


mse_values = []
mae_values = []
results = []
for i in range(1, 15):
    # Parametrer le KNNImputer
    imputer = KNNImputer(n_neighbors=i,weights='uniform')

    # Remplacer les NaN
    imputed_df = pd.DataFrame(imputer.fit_transform(df_cc), columns=df_cc.columns)


    # Calcul MSE et MAE entre imputed_df and original df1 sans NaN
    mse = mean_squared_error(df1.values, imputed_df.values)
    mae = mean_absolute_error(df1.values, imputed_df.values)

    
    mse_values.append(mse)
    results.append((i, mse, mae))
    
print("n_neighbors, MSE, MAE")
for result in results:
    print(*result)
    
    
# Plotting the MSE and MAE values
plt.plot(range(1, 15), mse_values, label='MSE')
plt.xlabel('n_neighbors')
plt.ylabel('Error')
plt.legend()
plt.show()

"""*********************************************************************************"""

# KNNImputer => Choix de n_neighbors=14
imputer = KNNImputer(n_neighbors=14,weights='uniform')

# Remplaçer les valeurs manquantes
imputed_df = pd.DataFrame(imputer.fit_transform(df_cc), columns=df_cc.columns)


# Calcul MSE et MAE entre imputed_df et df1 sans valeurs manquantes
mse = mean_squared_error(df1.values, imputed_df.values)
mae = mean_absolute_error(df1.values, imputed_df.values)
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print("\n")


# Représentation graphique des distributions
plt.figure(figsize=(20,35))
for i, col in enumerate(imputed_df.columns):
    if imputed_df[col].dtype != 'object':
        ax = plt.subplot(9, 3, i+1)
        sns.kdeplot(imputed_df[col], ax=ax)
        plt.xlabel(col)
        
plt.show()



# Box-plot 
# Avant traitement des outliers
# Set the figure size
plt.figure(figsize=(20, 10))

# Boucle au sein des colonnes du dataframe
for i, column in enumerate(imputed_df.columns):
    plt.subplot(5, 5, i+1)
    sns.boxplot(data=imputed_df[column], orient='v')
    plt.title(f"Box plot of {column}")

# Affiche du box-plot
plt.tight_layout()
plt.show()

# Gestion des outliers
from scipy import stats

# # Winsorization
# for col in imputed_df.columns:
#     imputed_df[col] = stats.mstats.winsorize(imputed_df[col], limits=0.05)


from scipy import stats
# Log Transformation
for col in imputed_df.columns:
    imputed_df[col] = np.log(imputed_df[col]+1)


# Checker la corrélation

plt.figure(figsize=(20,15))
sns.heatmap(imputed_df.corr(), annot=True,cmap='Blues')
plt.show()


"""*********************************************************************************"""
# New test

"""PCA is sensitive to the scale of the variables, so standardizing the data prior to applying PCA 
can ensure that the magnitude of the variables does not unduly influence the results. 
This can be particularly important if the variables have very different scales.
"""
scaler=StandardScaler()
df_scaler=scaler.fit_transform(imputed_df)

# ACP
pca = PCA(n_components=2, random_state=1)
data_pca = pca.fit_transform(df_scaler)

# Visualiser l'ACP
plt.figure(figsize=(15,10))
plt.scatter(data_pca[:, 0], data_pca[:, 1], s=15)
plt.show()


"""
K-means clustering, on the other hand, is sensitive to the relative scale of the variables, 
so normalizing the data can help ensure that all variables contribute equally to the distances 
used to form the clusters.
"""
# Normalisation pour les kmeans
df_norm=normalize(data_pca) 
df_norm=pd.DataFrame(df_norm)

scores = []
for k in range(2,15):
    km = KMeans(n_clusters=k,random_state=1)
    km = km.fit(df_norm)
    scores.append(km.inertia_)
dfk = pd.DataFrame({'Cluster':range(2,15), 'Score':scores})
plt.figure(figsize=(8,5))
plt.plot(dfk['Cluster'], dfk['Score'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

for i in range(2,15):
    kmeans_labels=KMeans(n_clusters=i,random_state=1).fit_predict(df_norm)
    print("Silhouette score for {} clusters k-means : {} ".format(i,metrics.silhouette_score(df_norm,kmeans_labels, 
                                                                                             metric='euclidean').round(3)))

     
# on conserve le max pour silhouette => 5 clusters
kmeans_labels=KMeans(n_clusters=5,random_state=1).fit_predict(df_norm)

# Calcul des métriques
sil_score= metrics.silhouette_score(df_norm,kmeans_labels, metric='euclidean').round(3)
ch_score = calinski_harabasz_score(df_norm, kmeans_labels)
db_score = metrics.davies_bouldin_score(df_norm, kmeans_labels)

print("Score de Silhouette score :", sil_score)
print("Score de Calinski-Harabasz :", ch_score)
print("Score de Davies-Bouldin :", db_score)

"""La métrique de Calinski-Harabasz est utilisée pour évaluer la qualité d'un clustering en comparant la variance intra-cluster à la variance inter-cluster. Plus précisément, elle mesure la ratio entre la variance de la somme des distances intra-cluster et la variance de la somme des distances inter-cluster.

Plus le score de Calinski-Harabasz est élevé, plus les clusters sont homogènes à l'intérieur et hétérogènes entre eux. Cela signifie que les données sont bien divisées en groupes distincts.

Une faible valeur de Calinski-Harabasz peut indiquer que les clusters sont mélangés, ce qui peut rendre difficile la classification et la compréhension des données.

Il est important de noter que la métrique de Calinski-Harabasz peut être influencée par le nombre de clusters choisi. Il est donc souvent utilisé en combinaison avec d'autres métriques pour déterminer le nombre optimal de clusters.
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans

# 2D
plt.figure(figsize=(8,5))
plt.scatter(data_pca[:,0], data_pca[:,1], c=kmeans_labels)
plt.title("KMeans Clustering Results in 2D")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()

# pas à mettre
plt.figure(figsize=(8,5))
plt.scatter(df_norm.iloc[:, 0], df_norm.iloc[:, 1], c=kmeans_labels)
plt.title("KMeans Clustering Results in 2D")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()


"""*********************************************************************************"""

from sklearn.decomposition import NMF

# Converti df to array
X_NMF = imputed_df.values

# NMF model
model = NMF(n_components=2, init='random', random_state=42)

# Fit le modèle 
fit_NMF = model.fit_transform(X_NMF)

# Normaliser pour les kmeans
df_norm_NMF=normalize(fit_NMF) 
df_norm_NMF=pd.DataFrame(df_norm_NMF)

scores = []
for k in range(2,15):
    km = KMeans(n_clusters=k,random_state=42)
    km = km.fit(df_norm_NMF)
    scores.append(km.inertia_)
dfk = pd.DataFrame({'Cluster':range(2,15), 'Score':scores})
plt.figure(figsize=(8,5))
plt.plot(dfk['Cluster'], dfk['Score'], marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

for i in range(2,15):
    kmeans_labels=KMeans(n_clusters=i,random_state=42).fit_predict(df_norm_NMF)
    print("Silhouette score for {} clusters k-means : {} ".format(i,metrics.silhouette_score(df_norm_NMF,
                                                                                             kmeans_labels, 
                                                                                             metric='euclidean').round(3)))

# K-means
kmeans = KMeans(n_clusters=5, random_state=42)

# Ajuster le modèle KMeans à la matrice factorisée obtenue à partir de NMF
kmeans_labels = kmeans.fit_predict(df_norm_NMF)

# Calcul des métriques
sil_score_NMF= metrics.silhouette_score(df_norm_NMF,kmeans_labels, metric='euclidean').round(3)
ch_score_NMF = calinski_harabasz_score(df_norm_NMF, kmeans_labels)
db_score_NMF = metrics.davies_bouldin_score(df_norm_NMF, kmeans_labels)

print("Score de Silhouette score :", sil_score_NMF)
print("Score de Calinski-Harabasz :", ch_score_NMF)
print("Score de Davies-Bouldin :", db_score_NMF)



import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn.cluster import KMeans


# 2D
plt.figure(figsize=(8,5))
plt.scatter(fit_NMF[:,0], fit_NMF[:,1], c=kmeans_labels)
plt.title("KMeans Clustering Results in 2D")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.show()

"""*********************************************************************************"""
# Profil NMF + kmeans
imputed_df['Clusters']=list(kmeans_labels)
customers=pd.DataFrame(imputed_df['Clusters'].value_counts()).rename(columns={'Clusters':'Number of Customers'})
customers.T

means=pd.DataFrame(imputed_df.describe().loc['mean'])
means.T.iloc[:,[0,1,6,8,9,11,12,16]].round(1)

imputed_df.set_index('Clusters')
grouped=imputed_df.groupby(by='Clusters').mean().round(1)
grouped.iloc[:,[0,1,6,8,9,11,12,16]]

features=["BALANCE","BALANCE_FREQUENCY","PURCHASES_FREQUENCY","PURCHASES_INSTALLMENTS_FREQUENCY","CASH_ADVANCE_FREQUENCY","PURCHASES_TRX","CREDIT_LIMIT","TENURE"]
plt.figure(figsize=(25,20))
for i,j in enumerate(features):
    plt.subplot(3,3,i+1)
    sns.barplot(grouped.index,grouped[j])
    plt.title(j,fontdict={'color':'darkblue'})
plt.tight_layout()
plt.show()


"""*********************************************************************************"""
# Spectral Clustering (fonctionne pas ?)

scaler=StandardScaler()
df_scaler=scaler.fit_transform(imputed_df)

# Normaliser pour les kmeans
df_norm_SC=normalize(df_scaler) 


# We can apply both (StandartScaler and Normalize) on our data before clustering. 
df_norm_SC=pd.DataFrame(df_norm_SC)

silhouette_list_spectral= []

for cluster in range(2,10):
    for neighbours in np.arange (3,10,2):
        spectral = SpectralClustering(n_clusters=cluster, affinity="nearest_neighbors",n_neighbors=neighbours, assign_labels='discretize',
                                      random_state=42).fit_predict(df_scaler)
        sil_score = metrics.silhouette_score(df_scaler,spectral, metric='euclidean')
        silhouette_list_spectral.append((cluster,sil_score, neighbours))

    
df_spectral= pd.DataFrame(silhouette_list_spectral, columns=['cluster', 'sil_score', 'neighbours'] )

df_spectral.sort_values('sil_score', ascending= False)
"""*********************************************************************************"""
















