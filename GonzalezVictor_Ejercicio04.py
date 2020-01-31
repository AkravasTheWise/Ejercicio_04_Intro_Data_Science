#!/usr/bin/env python
# coding: utf-8

# In[34]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.preprocessing 
#preprocessing estandariza los datos

import itertools

get_ipython().run_line_magic('matplotlib', 'inline')


# # Lectura de datos

# In[2]:


data = pd.read_csv('Cars93.csv')


# In[3]:


data[:2]


# # Selección de target y predictores

# In[4]:


print(data.keys())
#keys imprime los nombres de las columnas


# In[5]:


Y = np.array(data['Price'])
columns = ['MPG.city', 'MPG.highway', 'EngineSize', 'Horsepower', 'RPM', 'Rev.per.mile', 
          'Fuel.tank.capacity', 'Length', 'Width', 'Turn.circle', 'Weight']
X = np.array(data[columns])


# In[6]:


print(np.shape(Y), np.shape(X))


# # Renormalización de los datos para que todas las variables sean comparables y elección de training y test

# In[40]:


scaler = sklearn.preprocessing.StandardScaler()
#importo la clase para renormalizar los datos
scaler.fit(X)
#hago el fit escalado
X_scaled = scaler.transform(X)
#transformo los X para poder compararlos entre sí

X_train, X_test, Y_train, Y_test = sklearn.model_selection.train_test_split(X_scaled, Y, test_size=0.5)


# In[ ]:


R_cuadrados=[]
num_variables=[]
for i in range(11):
    #genero los predictores aleatoriamente
    predictores=list(itertools.permutations(np.arange(11),i+1))
    mejor_fit=1
    mejores_predictores=[]
    for j in predictores:
        X_fit=X_train[:,j]
        regresion = sklearn.linear_model.LinearRegression()
        regresion.fit(X_fit, Y_train)
        if (regresion.score(X_test[:,j],Y_test) < mejor_fit):
            mejor_fit=regresion.score(X_test[:,j],Y_test)
            mejores_predictores=regresion.coef_

    R_cuadrados.append(mejor_fit)
    num_variables.append(i+1)


# In[11]:


plt.plot(num_variables,R_cuadrado)
plt.show()
plt.savefig('R_cuadrado_vs_lambda')


# In[ ]:




