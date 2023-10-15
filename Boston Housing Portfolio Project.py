#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Data Source: U.S. Census Services Housing in Boston Masschusetts


# In[1]:


# Import pandas
import pandas as pd 


# In[11]:


# Read data into pandas dataframe. The "pd.read_csv" function was used to read the boston.csv file into the IDE 
# (slashes within the path must face the correct direction "/").
boston = pd.read_csv('C:/Users/melis/OneDrive/Documents/boston.csv')


# In[11]:


# Show first five rows 
boston.head(5)


# In[12]:


# Show dimensions. The "shape method" from pandas used to see the number of rows and columns. 
# There are 506 rows and 13 columns.
boston.shape


# In[14]:


# The "dtypes" method returnns the data type for each variable (integer V.S. categorical).
boston.dtypes


# In[15]:


# The "CHAS" variable represents the Charles River dummy variable & should be categorical so a conversion is needed 
# to convert this "CHAS" variable from integer to categorical type. To do this, the "as type" method is used.
# (The CHAS integer output for CHAS was see in a previous line that was deleted).
boston["CHAS"]=boston ["CHAS"].astype ("category")


# In[17]:


# The "describe" function was used to get the summary statistics information for the variables. 
# The output of the describe function shows the count for each variable (which is the number of rows in the dataset).
boston.describe ()


# In[18]:


# The minimum value for CRIM - per capita crime rate by town
boston['CRIM'].min()


# In[19]:


# The maximum value for CRIM - per capita crime rate by town 
boston['CRIM'].max()


# In[22]:


# The average value for CRIM - per capita crime rate by townboston
boston['CRIM'].mean()


# In[26]:


# Select first row. To select individual data or slices of a DataFrame, the "loc" or "iloc" 
# function can be used. For this function, the space before the comma represents the integer position for the rows, 
# and the space after the comma represents the integer position for the columns. Since the integer position in Python 
# starts from zero, to select the 1st row we must put a zero in the integer position for the rows, which is 
# the space before the comma. This example returned the 1st row in the boston DataFrame.
boston.iloc [0, ]


# In[27]:


# Select three rows (row 2-4) and first column. Here, the "iloc" function is used to select the 3 rows starting from 
# row number 2 to row number 4 & the 1st column.To select the 1st column zero was used for the column position. 
# The colon represents that we are going to select a continuous number of rows & it is inclusive of the number 
# before the colon & the number after the colon is exclusive. So, we exclude the the row before row number 5. 
# Since 4 represents row 5, it is rows 2-3 that will show. 
boston.iloc[1:4,0]


# In[28]:


# Select three rows (row 1-4) and all columns. In the brackets of "iloc", for the integer position of the rows, 
# we can put a colon & a "4" before the comma to represent that we want to go from the 1st to the 4th row. 
# The colon after the comma (in the integer position of the columns) indicates that we want to select all the columns. 
boston.iloc[:4,:]


# In[8]:


# Import pandas
import pandas as pd 


# In[13]:


# Show top 5 rows where column "CHAS" is 1. This method can be used when we are interested in selecting the slice of the 
# dataset with a specific value for some of the variables.
boston[boston['CHAS']==1].head(5)


# In[14]:


# Select rows where column "CHAS" is 1 AND "RM" is larger than 2.we can combine several conditions to help filter 
# data that we are interested in. Here, we are interested in the CHAS variable equal to 1, more than 6 rooms,
# & crime rate less than 1. So, this line returns a subset of the data. 
boston[(boston['CHAS']==1) & (boston['RM']>6) & (boston['CRIM']<1)]


# In[1]:


# Import pandas
import pandas as pd 


# In[3]:


# Read data into pandas dataframe
boston = pd.read_csv('C:/Users/melis/OneDrive/Documents/boston.csv')


# In[4]:


# Import libraries
import numpy as np
from sklearn import preprocessing


# In[ ]:


# Create a Min-Max scaler to rescale values of numerical features to between two values. 
minmax_scaler = preprocessing.MinMaxScaler (feature_range = (0,1))


# In[7]:


# Select numerical variable. Once we create a MinMax scaler we can apply the scaler to the numerical variables 
# in the boston data. So, we need to select all the numerical variables in the Boston DataFrame. To do this we 
# can use the function called "select dtypes" & we specify that we want to select all the numerical variables 
# (np.number). We save all the numerical variables in this boston numerical object.  
boston_numerical = boston.select_dtypes(include=np.number)


# In[9]:


# Scale features. The MinMax scaler is applied to the numerical variables. So, we use the "fit_transform" method to 
# apply this MinMax scaler to the numerical variables that have been saved in the bostonnumerical 
# object in the parenthesis. Then, we save the values after we perform the MinMax scaling to the sclaed Boston object. 
# So, it is an array & we can see the output of the scaled Boston array.   
scaled_boston = minmax_scaler.fit_transform(boston_numerical)
scaled_boston


# In[10]:


# Create a dataframe to save scaled features. In the last line, the "describe function" to take a look at the 
# summary statistics of the data frame after the MinMaxScalings, we see that all the numerical variables 
# have a min value of zero & max value of 1. 
scaled_boston_df= pd.DataFrame (scaled_boston, columns = boston_numerical.columns)

scaled_boston_df.describe()


# In[11]:


# Standardize & transform a feature to have a mean of 0 & a standard deviation of 1
scaler=preprocessing.StandardScaler()


# In[12]:


# Transform features. We use the fit_transform method to apply this standard scaler to the numerical variables 
# in the Boston DataFrame. The values are saved in this standardization to in this Boston Standardized Object. 
boston_standardized = scaler.fit_transform(boston_numerical)


# In[13]:


# We can also create a dataframe to save a standardized features.So, we save the standardized features in this 
# boston dataframe object. We can also take a look at the summary statistics for the standardized features 
# and we may notice that the mean value for all the numerical variable is almost zero & the standard deviation 
# for all the numerical variables is about 1. 
standardized_boston_df = pd.DataFrame(boston_standardized, columns=boston_numerical.columns)

standardized_boston_df.describe()


# In[14]:


# NAN: not a number. If we take a look at the first 10 rows in the Boston dataframe we, will notice that some of 
# the values have NaN which means “Not A Number.” These kind of values are the missing values in our dataframe.
boston. head(10)


# In[15]:


# The functions called “isnull” & “sum” are used to help count how many missing values are in each of the 
# columns in the Boston dataframe. It can be seen, most of the columns don't have missing values except 
# the one which is the variable of “CHAS”.
boston.isnull().sum()


# In[16]:


# Fill Nan with 0 (or with the averge). The CHAS variable has two values: 1 or 0. 
# The desired result is to fill the 3 missing values with a 0 by using the "fillna" method. 
boston['CHAS']=boston['CHAS'].fillna(0) 


# In[17]:


# Use isnull() to check missing values, now there are no missing values. 
print(boston[['CHAS']].isnull().sum())


# In[18]:


# Drop rows with missing values. 
# A more direct way for dealing with missing values is to delete them. For this we can use the "dropna" function 
# to drop any rows with missing values. Since, we do not want a copy of the missing values, we need to set an "inplace" 
# parameter equal to True. Then, using the "Print Shape" method to see that the number of rows in the boston dataframe 
# after deletinng the 3 rows with the missing values.
boston.dropna(inplace=True)
print(boston.shape)


# In[19]:


# Import libraries for data visualization. 
import seaborn as sns 


# In[20]:


# Histogram to show the distribution of MDEV. The color parameter is set to red. 
sns.histplot(boston['MDEV'], color = 'red')


# In[23]:


# Histogram to show the distribution of MDEV. The color parameter is set to turquoise. The histogram shows the 
# ditribution of the variable, MDEV, which is the price of housing.
sns.histplot(boston['MDEV'], color = 'turquoise')


# In[27]:


# Boxplot of the of MDEV. A Box Plot can be used to show the basic summary statistics for a numerical variable 
# by means of the Box Plot function. Notice, 50% have a median house value between 17-25. 
# Also, there are outliers in this house value variable. We may need to think about how to deal with these 
# kinds of outliers if we want to perform some data mining tasks. 
sns.boxplot(y='MDEV', data=boston)


# In[28]:


# Boxplot of the MDEV variable (MDEV) across the Charles River dummy variable (CHAS). Here, the line returns a 
# side-by-side Box Plot for numerical variables across different conditions. This example allows one to analyze if 
# the house values are different between the tracks that bound the Charles River & do not bound the Charles River.  
sns.boxplot(x='CHAS' ,y= 'MDEV',data=boston)


# In[30]:


# A scatter plot can help us visualize the general relationship between 2 numerical variables. 
# Line 30 returns a scatterplot to visualize the general distribution of house values (MDEV) 
# against the low status rate(LSTAT). We can also use the scatter plot to perform preliminary visualization 
# analysis to explore the relationships b/w the variables in the dataframe & we can use some data mining models 
# to do the significance test for these relationships. 
sns.scatterplot(x='LSTAT',y='MDEV', data=boston, legend=False) 

