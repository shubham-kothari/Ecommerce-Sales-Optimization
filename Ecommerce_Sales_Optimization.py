#!/usr/bin/env python
# coding: utf-8

# An Ecommerce company based in New York City that sells clothing online but they also have in-store style and clothing advice sessions. Customers come in to the store, have sessions/meetings with a personal stylist, then they can go home and order either on a mobile app or website for the clothes they want.
# 
# The company is trying to decide whether to focus their efforts on their mobile app experience or their website. 
# 
# It's a fake Dataset.

# 
# * Email: Email of the Customer.
# * Address: Address of the Customer.
# * Avatar: Avatar or Alias of the Customer.
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 
# * Yearly Amount Spend: Total amount of money spend by the customer yearly.

# ## Importing the libraries 
# 
# ** Import pandas, numpy, matplotlib,and seaborn **

# In[83]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Reading the Data
# 
# The Ecommerce Customers csv file from the company. 
# It has Customer info, such as Email, Address, and their color Avatar. Then it also has numerical value columns:
# 
# * Email: Email of the Customer.
# * Address: Address of the Customer.
# * Avatar: Avatar or Alias of the Customer.
# * Avg. Session Length: Average session of in-store style advice sessions.
# * Time on App: Average time spent on App in minutes
# * Time on Website: Average time spent on Website in minutes
# * Length of Membership: How many years the customer has been a member. 
# * Yearly Amount Spend: Total amount of money spend by the customer yearly.
# 
# **Reading in the Ecommerce Customers csv file as a DataFrame called df.**

# In[84]:


df = pd.read_csv("Ecommerce Customers")


# **Checking out the head of Ecommerce Customers, and checking out its info and descriptive statistic**

# In[85]:


df.columns


# In[86]:


df.info()


# In[87]:


df.describe()


# ## Exploratory Data Analysis
# 
# **Let's explore the data!**
# 
# **Using seaborn to create a jointplot to compare the Time on Website and Yearly Amount Spent columns.**

# **Plotting seaborn's pairplot to explore the types of relationships across the entire data set.**

# In[88]:


sns.set_style('whitegrid')
sns.pairplot(df,palette="GnBu_d")


# In[89]:


plt.figure(figsize=(10,6))
sns.heatmap(df.corr(),cmap='inferno',annot=True,)


# 
# **Creating a linear model plot (using seaborn's lmplot) of  Yearly Amount Spent vs. Length of Membership.**

# In[91]:


sns.lmplot('Length of Membership','Yearly Amount Spent',df,aspect=1.5)


# **We can easily analyse that Length of Membership and Yearly Amount Spent are postively Coorelated means higher the Legth of Membershiphigher will be the spending of the customer.**

# ** Doing the same thing but with the Time on App column instead. **

# In[92]:


sns.lmplot('Time on App','Yearly Amount Spent',df,aspect=1.5)


# **We can easily analyse that Time on App and Yearly Amount Spent are postively Coorelated means higher the time spend on app higher will be the spending of the customer.**

# In[93]:


sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=df,height=8)


# **Finding the top 3 most popular Avatars.**

# In[133]:


df['Avatar'].value_counts().head(3)


# **Finding the top 3 most popular email providers/hosts.**

# In[94]:


df.head()


# In[132]:


df['Email'].apply(lambda x: x.split('@')[1]).value_counts().head(3)


# # Preparing the data
# 

# In[95]:


df['Avatar'].value_counts().head(3)


# In[97]:


X = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
y = df['Yearly Amount Spent']


# **Normalizing the 'Avg. Session Length',  'Time on App',  'Time on Website',  'Length of Membership' columns.**

# **Import StandardScaler from sklearn.preprocessing**

# In[98]:


from sklearn.preprocessing import StandardScaler


# In[99]:


SS = StandardScaler()
SS.fit(X)
scaled = SS.transform(X)


# In[100]:


X = pd.DataFrame(scaled,columns=[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']])


# ## Training and Testing Data
# 
# Now that we've explored the data a bit, let's go ahead and split the data into training and testing sets.
# ** Set a variable X equal to the numerical features of the customers and a variable y equal to the "Yearly Amount Spent" column. **

# ** Use model_selection.train_test_split from sklearn to split the data into training and testing sets. Set test_size=0.3 and random_state=101**

# In[102]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)


# ## Training the Model
# 
# Now its time to train our model on our training data!
# 
# ** Import LinearRegression from sklearn.linear_model **

# In[103]:


from sklearn.linear_model import LinearRegression


# **Creating an instance of a LinearRegression() model named lm.**

# In[104]:


lm = LinearRegression()


# ** Train/fit lm on the training data.**

# In[105]:


lm.fit(X_train,y_train)


# **Printing out the coefficients of the model**

# In[106]:


# The coefficients
print('Coefficients: \n', lm.coef_)


# In[107]:


# The coefficients
print('Coefficients: \n', lm.coef_)


# ## Predicting Test Data
# After fitting our model, let's evaluate its performance by predicting off the test values!
# 
# ** Using lm.predict() to predict off the X_test set of the data.**

# In[108]:


predictions = lm.predict( X_test)


# ** Creating a scatterplot of the real test values versus the predicted values. **

# In[110]:


plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# ## Evaluating the Model
# 
# Let's evaluate our model performance by calculating the residual sum of squares and the explained variance score (R^2).
# 
# ** Calculate the Mean Absolute Error, Mean Squared Error, and the Root Mean Squared Error. Refer to the lecture or to Wikipedia for the formulas**

# In[111]:


# calculate these metrics by hand!
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


# ## Residuals
# 
# You should have gotten a very good model with a good fit. Let's quickly explore the residuals to make sure everything was okay with our data. 
# 
# **Plot a histogram of the residuals and make sure it looks normally distributed. Use either seaborn distplot, or just plt.hist().**

# In[131]:


plt.figure(figsize=(10,5))
sns.distplot((y_test-predictions),bins=30)


# ## Conclusion
# Let's interpret the coefficients to get an idea to the answer the question, where do we need to focus our efforts, on mobile app or on website development? 

# In[113]:


coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients


# Interpreting the coefficients:
# 
# - Holding all other features fixed, a 1 unit increase in **Avg. Session Length** is associated with an **increase of 25.76 total dollars spent**.
# - Holding all other features fixed, a 1 unit increase in **Time on App** is associated with an **increase of 38.33 total dollars spent**.
# - Holding all other features fixed, a 1 unit increase in **Time on Website** is associated with an **increase of 0.19 total dollars spent**.
# - Holding all other features fixed, a 1 unit increase in **Length of Membership** is associated with an **increase of 61.7 total dollars spent**.

# **This is tricky, there are three ways to think about outcomes from this**

# 
# **1. Develop the Website to catch up to the performance of the mobile app.**
# 
# **2. Or, Develop the app more since that is what is working better.**
# 
# **3. Or, We can focus on Length of Membership since it correlate most with the yearly amount spend.**
