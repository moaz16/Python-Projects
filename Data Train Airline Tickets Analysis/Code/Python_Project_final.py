#!/usr/bin/env python
# coding: utf-8

# # 1- Importing Libraries

# In[41]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#plt.style.use('seaborn') #ggplot
from pandas import ExcelWriter
from pandas import ExcelFile
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler


# # 2- Load Data

# In[42]:


flight = pd.read_excel('Data_Train_Airline_Tickets.xlsx')
flight


# # 3- Data Exploration

# In[43]:


flight.head(10)


# In[44]:


flight.tail(10)


# In[45]:


flight.info() # general information about data


# In[46]:


flight.columns # show columns


# In[47]:


flight.dtypes # show columns data type


# In[48]:


flight.isna().sum()


# In[49]:


flight.isnull() # returns if the cell is null or not


# In[50]:


flight['Price'].unique() # check every column if there is a value instead of null.


# # 4- Data Cleansing

# (1) remove duplicates

# In[51]:


flight.duplicated().sum() # show how many duplicated rows


# In[52]:


flight = flight.drop_duplicates() #remove duplicates


# In[53]:


flight.duplicated().sum() # show how many duplicated rows


# In[54]:


flight.isnull().sum()


# In[55]:


flight = flight.drop('Route',axis=1) #route is already an useless column because its values already exsisted in the other columns.
flight


# In[56]:


flight['Total_Stops'].value_counts() 


# In[57]:


flight['Total_Stops'] = flight['Total_Stops'].fillna(flight['Total_Stops'].mode()[0])


# In[58]:


flight.isnull().sum() # now our data is ready to analysis


# In[59]:


flight['Total_Stops'] = flight['Total_Stops'].replace({'non-stop':0, '1 stop':1, '2 stops':2, '3 stops':3, '4 stops':4})


# In[60]:


flight['Total_Stops'].unique()


# In[61]:


flight['Additional_Info'].unique()


# In[62]:


flight['Additional_Info'] = flight['Additional_Info'].replace({'No Info': 'Standard', 'No info':'Standard'})


# In[63]:


flight['Additional_Info'].unique()


# In[64]:


flight['Date_of_Journey'] = pd. to_datetime (flight['Date_of_Journey']) 
print (flight.dtypes)


# In[65]:


flight['year'] = flight['Date_of_Journey'].dt.year
flight


# In[66]:


flight['year'].unique() #we have only 2019 so its not important


# In[67]:


flight = flight.drop('year',axis=1) 


# In[68]:


flight['month'] = flight['Date_of_Journey'].dt.month
flight


# In[69]:


flight['day'] = flight['Date_of_Journey'].dt.day
flight


# In[70]:


flight = flight.drop('Date_of_Journey',axis=1)  #now we can drop the original column
flight


# In[71]:


flight['Arrival_Time'].unique()


# In[72]:


flight['Arrival_Time'] = pd. to_datetime (flight['Arrival_Time']) 


# In[73]:


flight['Arrival_Time'] = pd.to_datetime(flight['Arrival_Time'], format='%H:%M:%S').dt.hour
flight


# In[74]:


flight['Dep_Time'] = pd.to_datetime(flight['Dep_Time'], format='%H:%M').dt.hour
flight


# In[75]:


flight['Duration']=  flight['Duration'].str.replace("h", '*60').str.replace(' ','+').str.replace('m','*1').apply(eval)
flight


# # 5- EDA

# In[76]:


for feature in flight.columns[flight.dtypes != 'object']:
    plt.figure()
    plt.hist(flight[feature])
    plt.title(feature)
    plt.show()


# In[77]:


plt.figure(figsize = (15, 10))
plt.title('Count of flights per month')
ax=sns.countplot(x = 'month', data = flight)
plt.xlabel('Month')
plt.ylabel('Count of flights')
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',
                    color= 'black')


# In[78]:


plt.figure(figsize = (15, 10))
plt.title('Count of flights per month')
ax=sns.countplot(x = 'Dep_Time', data = flight)
plt.xlabel('Month')
plt.ylabel('Count of flights')
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',
                    color= 'black')


# In[79]:


plt.figure(figsize = (15, 10))
plt.title('Count of flights with different Airlines')
ax=sns.countplot(x = 'Airline', data = flight)
plt.xlabel('Airline')
plt.ylabel('Count of flights')
plt.xticks(rotation = 90)
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x()+0.25, p.get_height()+1), va='bottom',
                    color= 'black')


# In[80]:


plt.figure(figsize = (15, 10))
plt.title('Price VS Airlines')
plt.scatter(flight['Airline'], flight['Price'])
plt.xticks(rotation = 90)
plt.xlabel('Airline')
plt.ylabel('Price of ticket')
plt.xticks(rotation = 90)


# In[81]:


flight["Airline"].replace({'Multiple carriers Premium economy':'Other', 
                                                        'Jet Airways Business':'Other',
                                                        'Vistara Premium economy':'Other',
                                                        'Trujet':'Other'
                                                   },    
                                        inplace=True) # because its very low in count


# In[82]:


plt.figure(figsize = (15, 10))
plt.title('Price VS Additional Information')
sns.scatterplot(flight['Additional_Info'], flight['Price'],data=flight)
plt.xticks(rotation = 90)
plt.xlabel('Information')
plt.ylabel('Price of ticket')


# In[83]:


flight["Additional_Info"].value_counts()  #we can merge all small number of flights together


# In[84]:


flight["Additional_Info"].replace({'Change airports':'Other', 
                                                        'Business class':'Other',
                                                        '1 Short layover':'Other',
                                                        'Red-eye flight':'Other',
                                                        '2 Long layover':'Other',   
                                                   },    
                                        inplace=True)


# In[85]:


plt.figure(figsize = (15, 10))
plt.title('Price VS month')
sns.scatterplot(flight['month'], flight['Price'],data=flight)
plt.xticks(rotation = 90)
plt.xlabel('Information')
plt.ylabel('Price of ticket')


# In[86]:


plt.figure(figsize = (15, 10))
plt.title('Price VS dep_time')
sns.scatterplot(flight['Dep_Time'], flight['Price'],data=flight)
plt.xticks(rotation = 90)
plt.xlabel('Information')
plt.ylabel('Price of ticket')


# In[87]:


flight["Dep_Time"].replace({1:1, 2:1, 3:1, 4:1, 5:2, 6:2, 7:2, 8:2, 9:2, 10:2, 11:2, 12:2, 13:2, 14:2,
                            15:2, 16:3, 17:3, 18:3, 19:3, 20:3, 21:3, 22:3, 23:3, 0:3
                                                   },    
                                        inplace=True)
flight["Dep_Time"].unique() # we divided the dep time into three levels


# In[88]:


flight.head()


# In[89]:


flight.corr()


# In[90]:


plt.figure(figsize=(20,15))
sns.heatmap(flight.corr(), annot=True)


# In[91]:


flight.corr().unstack()


# In[92]:


feature_corr = flight.corr().unstack().sort_values()
feature_corr


# In[93]:


feature_corr[(feature_corr>=0.5)&(feature_corr<1)]


# In[94]:


print(feature_corr[(abs(feature_corr)>=0.5) & (abs(feature_corr)<1)].drop_duplicates())


# In[95]:


high_corr_df = pd.DataFrame(feature_corr[(abs(feature_corr)>=0.5) & (abs(feature_corr)<1)].drop_duplicates())
high_corr_df.index


# # Features Reduction

# In[96]:


flight = flight.drop(columns = ['Duration'])


# In[97]:


flight.corr()


# In[98]:


plt.figure(figsize=(20,15))
sns.heatmap(flight.corr(), annot=True)


# # Features Distribution

# In[99]:


plt.hist(flight['Price'])
plt.title('flight Price Distribution', size=16)
plt.show()


# In[100]:


flight.columns[flight.dtypes!='object']


# In[101]:


num_feature = flight.columns[flight.dtypes!='object']

def my_plot(feature):
    plt.hist(flight[feature])
    plt.title(feature)
    plt.show()
        
for i in num_feature:
    my_plot(i)


# # Categorical Features Transformation

# In[102]:


flight_cat = flight.select_dtypes(include=['object'])


# In[103]:


flight_cat.head()


# In[104]:


flight_clear = pd.get_dummies(flight_cat, drop_first=True)
flight_clear


# In[105]:


flight_clear.info()


# In[106]:


flight_clear = pd.get_dummies(flight, drop_first=True)


# In[107]:


flight_clear.info()


# In[108]:


flight_clear.head()


# # Divide Data into Train & Test

# In[109]:


x = flight_clear.drop('Price', axis=1)
y = pd.DataFrame(flight_clear['Price'])


# In[110]:


x.head()


# In[111]:


y.head()


# In[112]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)


# In[113]:


print(x_train.shape)
x_train.head()


# In[114]:


print(y_train.shape)
y_train.head()


# In[115]:


x_test.shape, y_test.shape


# # Numerical Features Scaling

# In[116]:


scaler_x = MinMaxScaler().fit(x_train)
scaler_y = MinMaxScaler().fit(y_train)


# In[117]:


x_train_sc = scaler_x.transform(x_train)
x_test_sc = scaler_x.transform(x_test)


# In[118]:


y_train_sc = scaler_y.transform(y_train)
y_test_sc = scaler_y.transform(y_test)


# In[119]:


y_train


# In[120]:


y_train_sc


# # Apply Linear Regression

# In[121]:


lr_model = LinearRegression()
lr_model.fit(x_train_sc, y_train_sc)
y_pred_sc = lr_model.predict(x_test_sc)


# In[122]:


y_test_sc


# In[123]:


y_pred_sc


# In[124]:


mae = mean_absolute_error(y_test_sc, y_pred_sc)
rmse = np.sqrt(mean_squared_error(y_test_sc, y_pred_sc))

print('MAE = ', mae.round(4))
print('RMSE = ', rmse.round(4))


# In[125]:


y_test_inv = scaler_y.inverse_transform(y_test_sc.reshape(-1,1))
y_pred_inv = scaler_y.inverse_transform(y_pred_sc.reshape(-1,1))

actual_mae = mean_absolute_error(y_test_inv, y_pred_inv)
actual_rmse = np.sqrt(mean_squared_error(y_test_inv, y_pred_inv))

print('Actual MAE = ', int(actual_mae))
print('Actual RMSE = ', int(actual_rmse))


# In[126]:


y_test_inv


# In[127]:


y_pred_inv


# In[128]:


from sklearn.metrics import r2_score
print(r2_score(y_test_inv,y_pred_inv))

