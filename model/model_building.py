#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np


## one way ANOVA test
import statsmodels
import statsmodels.api as sm
from statsmodels.formula.api import ols


## model
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix 


# In[63]:


data = pd.read_csv("withtax.csv")
data = data.rename(columns={"GEO.id2": "id2"})
data.head()


# In[64]:



subcolumns = ["id2","Establishments not operated for the entire year", "Establishments operated for the entire year", "totalSales", "totalIncome"]
sub_data = data[data.columns[-14:]]
sub_data2 = data[subcolumns]
data2 = sub_data2.merge(sub_data, left_on = sub_data2.index, right_on = sub_data.index)
data2 = data2.drop(['key_0'], axis=1) 
data2.head()


# 

# In[65]:


data2 = data2.rename(columns={"Establishments not operated for the entire year": "estNotEntireYear", "Establishments operated for the entire year":"estEntireYear"}) 
data2.head()


# In[ ]:





# In[ ]:





# In[66]:


column_names = list(data2.columns)
for i in range(1,len(column_names)): 
    column_i= column_names[i] 
    formula_str = 'id2' + '~' + column_i
    data_lm = ols(formula_str,data = data2).fit()
    table = sm.stats.anova_lm(data_lm, typ=2)
    print(table)


# In[67]:


## DROP LAST ONE
data2 = data2.drop(columns = ['AGI_STUB','A18425'])
data2.head()


# In[68]:


## 50 percent sampled
## in which 60 percent as train, 20 percent as validate, 20 percent as test
data2 = pd.DataFrame.dropna(data2)
n = int(len(data)*0.5)
data_sampled = data.sample(n)
n = len(data_sampled)
print(n)
train_data = data_sampled.sample(int(n*0.8))
print(len(train_data)) ## 80%
test_data = data_sampled[(data_sampled.index).isin(train_data.index) == False]
print(len(test_data)) ## 20%
validate_data = train_data.sample(int(n*0.2))
print(len(validate_data)) ## 20%
train_data = train_data[(train_data.index).isin(validate_data.index) == False]
print(len(train_data)) ## 60%


# In[69]:


############################ model fitting and testing ##########################


# In[70]:


x_train = train_data.iloc[:, 4:].to_numpy()
y_train = train_data.iloc[:, 0].to_numpy()
x_test = test_data.iloc[:, 4:].to_numpy()
y_test = test_data.iloc[:, 0].to_numpy()
x_val = validate_data.iloc[:, 4:].to_numpy()
y_val = validate_data.iloc[:, 0].to_numpy()


# In[71]:


from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
x_train = sc_X.fit_transform(x_train)
x_test = sc_X.fit_transform(x_test)
x_val = sc_X.fit_transform(x_val)


# In[ ]:





# In[72]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
clf = RandomForestClassifier(n_estimators = 5)
cross_val_score(clf, x_train, y_train, cv = 3)


# In[78]:


print(np.mean(np.array([0.41583725, 0.6137931 , 0.8099123 ])))


# In[85]:


rfc = RandomForestClassifier(n_estimators = 5)
rfc.fit(x_train,y_train)
print("acc: ",rfc.score(x_test,y_test))


# In[86]:


for name, importance in zip(column_names[1:], rfc.feature_importances_):
    print(name, "=", importance)


# In[ ]:




