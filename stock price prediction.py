
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[4]:


df3=pd.read_csv("file:///C:/Users/Avinash/Downloads/31-05-2017-TO-30-05-2019ACCALLN.csv",parse_dates=['Date'])


# In[5]:


df.columns


# In[8]:


df=df3[:495]


# In[9]:


print(df.shape)
print(df.head(10))
df.tail(15)


# In[91]:


df2=df3[495:]


# In[75]:


df2


# In[12]:


plt.figure(figsize=(16,5))
plt.plot(df.Date,df['High Price'],color='tab:red')
plt.plot(df.Date,df['Low Price'],color='tab:blue')
plt.show()


# In[13]:


plt.figure(figsize=(16,5))
plt.plot(df.Date,df['Average Price'])


# In[14]:


df1=pd.DataFrame()
df1=df.drop(['Symbol', 'Series', 'Prev Close', 'Open Price', 'Last Price', 'Close Price', 'Average Price',
       'Total Traded Quantity', 'Turnover', 'No. of Trades', 'Deliverable Qty',
       '% Dly Qt to Traded Qty'],axis=1)


# In[15]:


df1['year'] = [d.year for d in df1.Date]
df1['month'] = [d.strftime('%b') for d in df1.Date]
years = df1['year'].unique()


# In[16]:


for i,y in enumerate(years):
    print(i,y)


# In[17]:


from statsmodels.tsa.seasonal import seasonal_decompose

result_mul = seasonal_decompose(df1['High Price'], model='multiplicative',freq=12)
result_add = seasonal_decompose(df1['High Price'], model='additive',freq=12)

plt.figure(figsize=(16,5))
result_mul.plot().suptitle('Multiplicative Decompose')
result_add.plot().suptitle('Additive Decompose')
plt.show()


# In[53]:


df1


# In[18]:


from statsmodels.tsa.stattools import adfuller

result=adfuller(df1['High Price'],autolag='AIC')
print(f'adf stastics:{result[0]}')
print(f'p-value:{result[1]}')

for key,value in result[4].items():
    print(f'critical value:')
    print(f'{key},{value}')


# In[19]:


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

fig, axes = plt.subplots(3, 3, sharex=True)
axes[0, 0].plot(df1['High Price']); axes[0, 0].set_title('Original Series')
plot_acf(df1['High Price'], ax=axes[0, 1])
plot_pacf(df1['High Price'],ax=axes[0,2])

# 1st Differencing
axes[1, 0].plot(df1['High Price'].diff()); axes[1, 0].set_title('1st Order Differencing')
plot_acf(df1['High Price'].diff().dropna(), ax=axes[1, 1])
plot_pacf(df1['High Price'].diff().dropna(), ax=axes[1, 2])
# 2st Differencing
axes[2, 0].plot(df1['High Price'].diff().diff()); axes[2, 0].set_title('2st Order Differencing')
plot_acf(df1['High Price'].diff().diff().dropna(), ax=axes[2, 1])
plot_pacf(df1['High Price'].diff().diff().dropna(), ax=axes[2, 2])


# In[20]:


from statsmodels.tsa.arima_model import ARIMA


model = ARIMA(df1['High Price'], order=(1,2,2))
model_fit = model.fit(disp=0)
print(model_fit.summary())


# In[21]:


# Plot residual errors
residuals = pd.DataFrame(model_fit.resid)
fig, ax = plt.subplots(1,2)
residuals.plot(title="Residuals", ax=ax[0])
residuals.plot(kind='kde', title='Density', ax=ax[1])
plt.show()


# In[24]:


# Actual vs Fitted
model_fit.plot_predict(dynamic=False)
plt.show()


# In[25]:


# Forecast
fc, se, conf = model_fit.forecast(12, alpha=0.05)  # 95% conf

# Make as pandas series
fc_series = pd.Series(fc, index=df2.index)
lower_series = pd.Series(conf[:, 0], index=df2.index)
upper_series = pd.Series(conf[:, 1], index=df2.index)


# In[26]:


plt.figure(figsize=(12,5), dpi=100)
plt.plot(df1['High Price'], label='training')
plt.plot(fc_series, label='forecast')
plt.plot(lower_series)
plt.plot(upper_series)
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
plt.show()


# In[93]:


df2['High Price']=fc_series


# In[27]:


plt.plot(df['High Price'])
plt.plot(df['High Price'].diff(12))


# In[29]:


import pmdarima as pm

# Seasonal - fit stepwise auto-ARIMA
smodel = pm.auto_arima(df['High Price'], start_p=1, start_q=1,
                         test='adf',
                         max_p=3, max_q=3, m=12,
                         start_P=0, seasonal=True,
                         d=None, D=1, trace=True,
                         error_action='ignore',  
                         suppress_warnings=True, 
                         stepwise=True)


# In[37]:


df5=pd.read_csv("file:///C:/Users/Avinash/Downloads/13-06-2018-TO-12-06-2019ACCALLN.csv")


# In[43]:


df5.tail(15)


# In[41]:


tf=df5['High Price']
tf1=tf[237:]


# In[76]:


df2


# In[66]:





# In[92]:


k=495
j=237
for i in range(8):
    df2.loc[k,['High Price']]=tf1[j]
    j=j+1
    k=k+1


# In[102]:


df2=df2.drop(labels='Date',axis=1)
df2=df2.dropna(axis=0,how='all')


# In[106]:


# Forecast
n_periods = 24
fitted, confint = smodel.predict(n_periods=n_periods, return_conf_int=True)
index_of_fc = np.arange(len(df['High Price']), len(df['High Price'])+n_periods)
# make series for plotting purpose
fitted_series = pd.Series(fitted, index=index_of_fc)
lower_series = pd.Series(confint[:, 0], index=index_of_fc)
upper_series = pd.Series(confint[:, 1], index=index_of_fc)

# Plot
plt.plot(df['High Price'])
plt.plot(df2['High Price'])
plt.plot(fitted_series, color='darkgreen')
plt.fill_between(lower_series.index, 
                 lower_series, 
                 upper_series, 
                 color='k', alpha=.15)

plt.show()


# In[36]:


print(fitted_series)
print(lower_series)
print(upper_series)

