
# coding: utf-8

# ## SBU - MAR 570 - Fall 2019 - D. Wilks Textbook, 3d ed.
# ### Gui Larangeira
# #### Example 10.1

# In[19]:


import matplotlib.pyplot as plt
import requests
import pandas as pd
import os
import numpy as np
from scipy.linalg import inv, solve, det, norm
from numpy import linalg
import statistics as stats

get_ipython().run_line_magic('pwd', '')


# In[5]:


os.chdir("C:/Users/Glarange/Google Drive/Stony/MAR_570")
get_ipython().run_line_magic('pwd', '')


# In[6]:


df = pd.read_csv("C:/Users/Glarange/Google Drive/Stony/MAR_570/A1.csv")
X = df.drop(df.columns[0], axis=1)
X = X.values
X


# In[45]:





# %COVARIANCE
# %Consider the covariance between observations at the two stations in A.1
# 
# %Form a data matrix [X] [31x6] where rows are time and columns are obs
# X=[Ithicaprecip,IthicamaxT,IthicaminT,Cananprecip,CananmaxT,CananminT];
# 
# %Must demean [X] according to 10.4
#    Xdm  =    X   -  ones(31) * X/31; %10.29
# %[31x6] =  [6x31]  [31x31]   [31x6] 
# 
#  %Compute covariance matrix [S] according to 10.30
#    S  =    Xdm' *  Xdm/(31-1);
# %[6x6]=   [6x31]  [31x6]
# 
# % S =
# % 
# %    5.8994e-02  -4.4452e-02   9.4867e-01   3.9314e-02  -7.5290e-02   4.6662e-01
# %   -4.4452e-02   5.9516e+01   7.5433e+01   2.3323e-02   5.8070e+01   5.1697e+01
# %    9.4867e-01   7.5433e+01   1.8547e+02   6.1000e-01   8.1633e+01   1.1080e+02
# %    3.9314e-02   2.3323e-02   6.1000e-01   2.8106e-02  -1.9935e-02   2.7827e-01
# %   -7.5290e-02   5.8070e+01   8.1633e+01  -1.9935e-02   6.1847e+01   5.6119e+01
# %    4.6662e-01   5.1697e+01   1.1080e+02   2.7827e-01   5.6119e+01   7.7581e+01
# 

# In[7]:


X.shape
ones = np.ones((31,31))

X_dm = X - ones @ X/31

S = X_dm.transpose() @ X_dm/(31-1)
# The Covariance Matrix is
S


# #### Example 12.1

# In[122]:


x_1 = X[:,2] 
x_2 = X[:,5]
# de-mean
xp_1 = x_1 - stats.mean(x_1)
xp_2 = x_2 - stats. mean(x_2)
xp = np.column_stack((xp_1,xp_2))


# In[141]:


fig1, ax1 = plt.subplots()
ax1.set_title("Canan vs. Ithaca anomalies")
ax1.set_xlabel("Ithaca min. temp. anomaly, x_1p, deg. F")
plt.scatter(xp_1,xp_2)


# In[140]:


d = xp.shape[0]
ones = np.ones((d,d))

xt = xp - ones @ xp/d

St = xt.transpose() @ xt/(d-1)
# The Covariance Matrix is
St


# In[139]:


e = linalg.eig(St)
e
ev = e[0]
eigval_1 = ev[0]
eigval_2 = ev[1]
e = e[1]
e_1 = eigvec[:,0]
e_2 = eigvec[:,1]
u = e @ xt.transpose()
u


# In[126]:


print(eigval_1/(eigval_1 + eigval_2))
print(u.shape)
print(e_1)


# In[138]:


t = np.arange(1,32)
xpr_1 = e_1[0] * u[0] 
xpr_2 = e_1[1] * u[0]


# In[136]:


#plt.plot(XT_p1)
fig2, ax2 = plt.subplots()

ax2.set_title("Ithaca anomalies")
ax2.set_xlabel("Day")
ax2.set_ylabel("degree Fahrenheit")
ax2.plot(t, xp_1, label = 'actual')
ax2.plot(t, xpr_1, label = 'PCA reconstructed 1st term')
ax2.legend()
plt.ylim(-30,30)


# In[142]:


fig3, ax3 = plt.subplots()
ax3.set_title("Canan anomalies")
ax3.set_xlabel("Day")
ax3.set_ylabel("degree Fahrenheit")
ax3.plot(t, xp_2, label = 'actual')
ax3.plot(t, xpr_2, label = 'PCA reconstructed 1st term')
ax3.legend()
plt.ylim(-30,30)

