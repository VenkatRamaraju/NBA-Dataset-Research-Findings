#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[92]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
from plotly import __version__
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected = True)
cf.go_offline()
get_ipython().run_line_magic('matplotlib', 'inline')
MVPsteph = pd.read_csv("MVPseason")
preSteph = pd.read_csv("players_stats(08-09).csv")
current = pd.read_csv("players_stats(18-19).csv") # Loading Base Data and Libraries


# # 2008 metrics

# In[93]:


Attempts2008 = preSteph["3PA"].sum()
made2008 = preSteph["3P"].sum()
percent2008 = made2008 / Attempts2008


# # 2018 Metrics

# In[94]:


Attempts2018 = current["3PA"].sum()
made2018 = current["3P"].sum()
percent2018 = made2018 / Attempts2018


# # Increase Percentage over 10-year span

# In[95]:


increase = ((Attempts2018-Attempts2008) / (Attempts2008))*100 


# # Average threes attempted by centers before and after the decade

# In[96]:


preSteph[preSteph["Pos"] == "C"]["3PA"].mean()
current[current["Pos"] == "C"]["3PA"].mean() 


# # Plotting threes attempted per position, 2008 and 2018

# In[97]:


sns.swarmplot(x="Pos", y="3PA", data=preSteph,palette='rainbow')
plt.ylim(0,1000)
plt.title("3PA per position in 2008") # 2008 positions swarmplot


# In[98]:


sns.swarmplot(x="Pos", y="3PA", data=current,palette='rainbow')
plt.ylim(0,1000)
plt.title("3PA per position in 2018") # 2018 positions swarmplot


# # Point guard scoring and rebounding increases over 10-year span

# In[99]:


(current[current["Pos"] == "PG"]["TRB"].sum() - preSteph[preSteph["Pos"] == "PG"]["TRB"].sum()) / preSteph[preSteph["Pos"] == "PG"]["TRB"].sum() * 100
# Increase in PG TRB %


# In[100]:


(current[current["Pos"] == "PG"]["PTS"].sum() - preSteph[preSteph["Pos"] == "PG"]["PTS"].sum()) / preSteph[preSteph["Pos"] == "PG"]["PTS"].sum() * 100
# Increase in PG PTS %


# # Cumulative Statistics since 1980

# In[101]:


totalStats = pd.read_csv("totalLeague-80to19")


# In[102]:


totalStats


# In[103]:


totalStats = totalStats.reindex(index=totalStats.index[::-1]) # Reversing row order for better visuals


# # Linear regression prediction model (training, testing, fit, predictions)

# In[104]:


stephYearly = pd.read_csv("StephYearlyStats")
stephAvgPredict = pd.read_csv("stephAvgPredict") # Loading more data sets


# In[105]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
lm = LinearRegression() 
# Libraries and creating model


# In[106]:


X_train = stephYearly[['GS', 'MP', 'FG', 'FGA', 'FG%',
       '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
       'ORB', 'DRB', 'STL', 'BLK', 'TOV', 'PF']]
y_train = stephYearly[["TRB", "AST", "PTS"]] # Training data is his 5-year compiledd statistics

X_test =  stephAvgPredict # Test data is an average set, used for prediction


# In[107]:


X_test = stephAvgPredict[['GS', 'MP', 'FG', 'FGA', 'FG%',
       '3P', '3PA', '3P%', '2P', '2PA', '2P%', 'eFG%', 'FT', 'FTA', 'FT%',
       'ORB', 'DRB', 'STL', 'BLK', 'TOV', 'PF']] # Test set, used to predict major statistical categories


# In[108]:


lm.fit(X_train,y_train) # Fitting the model to the data set


# In[109]:


prediction = lm.predict(X_test) # Predicting based on adjusted coefficients


# In[110]:


for i in range(0, 3): 
    prediction[0][i] = (prediction[0][i])/(71.6) # Standardizing to per-game statistics


# # Final Prediction

# In[111]:


prediction

