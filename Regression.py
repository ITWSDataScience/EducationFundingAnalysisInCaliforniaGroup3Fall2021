# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 12:01:26 2021

@author: smithd30
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import SpectralClustering
#loading dataset
data = pd.read_excel("Education Dataset.xlsx")
df = pd.DataFrame(data, columns = ['District', 'County', 'Students', 'Graduation Rate', 'Cost'])

#Assemble lists of desired variables
funding_per = []
grad_rate = []
students = []

no_outliers_funding_per = []
no_outliers_grad_rate = []
no_outliers_students = []

outlier_districts = []

for i,r in df.iterrows():
  funding_per.append(int(r['Cost'])/r['Students'])
  grad_rate.append(r['Graduation Rate'])
  students.append(r['Students'])
  if r['Graduation Rate'] > 20 and int(r['Cost'])/r['Students'] < 1000000:
    no_outliers_funding_per.append(int(r['Cost'])/r['Students'])
    no_outliers_grad_rate.append(r['Graduation Rate'])
    no_outliers_students.append(r['Students'])
  else:
    outlier_districts.append(r['District'])

#adjusting to a numpy array
funding = np.zeros(len(no_outliers_funding_per))
grad = np.zeros(len(no_outliers_grad_rate))
stu = np.zeros(len(no_outliers_students))
for j in range(0,len(stu)):
    funding[j] = no_outliers_funding_per[j]
    grad[j] = no_outliers_grad_rate[j]
    stu[j] = no_outliers_students[j]
    
'''
edit here to change the axis and questions being asked
'''
x_var = stu
x_var_name = 'Number of Students'
y_var = funding
y_var_name = 'Funding per Student'
 
#finding the linear regression
x_var = x_var.reshape(-1,1)
model = LinearRegression()
model.fit(x_var, y_var)

#editting data for linear regression plot (removed)
xmax = max(x_var)
x = np.linspace(0, xmax, 1001)
y = np.zeros(1001)
for j in range(0, 1001):
    y[j] = model.coef_*x[j] +model.intercept_

plt.plot(x_var,y_var,'ro')
plt.xlabel(x_var_name)
plt.ylabel(y_var_name)
plt.title('Polynomial Regression')

#finding polynomial regression
x = x.reshape(-1,1)
poly = PolynomialFeatures(degree = 4)
x_poly = poly.fit_transform(x)
poly.fit(x_poly,y)
lin2 = LinearRegression()
lin2.fit(x_poly, y)
plt.plot(x, lin2.predict(poly.fit_transform(x)), color = 'red')
plt.show()



#cluster data fitting
cluster_data = np.column_stack((x_var,y_var))

#cluster analysis 1
#K means clustering
model = KMeans(n_clusters = 4)
# fit the model
model.fit(cluster_data)
# assign a cluster to each example
yhat = model.predict(cluster_data)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	plt.scatter(cluster_data[row_ix, 0], cluster_data[row_ix, 1])
# show the plot
plt.title('K Means Cluster')
plt.xlabel(x_var_name)
plt.ylabel(y_var_name)
plt.show()



#cluster analysis 2
#affinity propagation
model = AffinityPropagation(damping=0.9)
# fit the model
model.fit(cluster_data)
# assign a cluster to each example
yhat = model.predict(cluster_data)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	plt.scatter(cluster_data[row_ix, 0], cluster_data[row_ix, 1])
# show the plot
plt.title('Affinity Propagation Cluster')
plt.xlabel(x_var_name)
plt.ylabel(y_var_name)
plt.show()



#cluster analysis 3
#agglomerative clustering
model = AgglomerativeClustering(n_clusters=4)
# fit model and predict clusters
yhat = model.fit_predict(cluster_data)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	plt.scatter(cluster_data[row_ix, 0], cluster_data[row_ix, 1])
# show the plot
plt.title('Agglomerative Cluster')
plt.xlabel(x_var_name)
plt.ylabel(y_var_name)
plt.show()



#cluster analysis 4
#BIRCH clustering
# define the model
model = Birch(threshold=0.01, n_clusters=4)
# fit the model
model.fit(cluster_data)
# assign a cluster to each example
yhat = model.predict(cluster_data)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	plt.scatter(cluster_data[row_ix, 0], cluster_data[row_ix, 1])
# show the plot
plt.title('BIRCH Cluster')
plt.xlabel(x_var_name)
plt.ylabel(y_var_name)
plt.show()



#cluster analysis 5
#DBSCAN
# define the model
model = SpectralClustering(n_clusters=4)
# fit the model
model.fit(cluster_data)
# assign a cluster to each example
yhat = model.fit_predict(cluster_data)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	plt.scatter(cluster_data[row_ix, 0], cluster_data[row_ix, 1])
# show the plot
plt.title('SpectralClustering Cluster')
plt.xlabel(x_var_name)
plt.ylabel(y_var_name)
plt.show()

