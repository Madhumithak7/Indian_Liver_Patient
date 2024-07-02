#!/usr/bin/env python
# coding: utf-8

# In[53]:


import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import plotly.graph_objects as go
import sklearn
from sklearn import preprocessing
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score , KFold
from sklearn.metrics import mean_squared_error , mean_absolute_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[54]:


df=pd.read_csv("Indian Liver Patient Dataset (ILPD).csv")
df.head()


# In[55]:


# Define the mapping from current column names to new column names
# Replace 'Current_Column_Name' with the actual column names in your CSV
column_mapping = {
    '65': 'age',
    'Female': 'gender',
    '0.7': 'Total_Bilirubin',
    '0.1': 'Direct_Bilirubin',
    '187': 'Alkaline_Phosphotase',
    '16': 'Alamine_Aminotransferase',
    '18': 'Aspartate_Aminotransferase',
    '6.8': 'Total_Protiens',
    '3.3': 'Albumin',
    '0.9': 'Albumin_and_Globulin_Ratio',
    '1': 'Dataset'
}

# Rename the columns
df = df.rename(columns=column_mapping)

# Display the DataFrame to verify the changes
print(df)


# In[56]:


print("number of variables",df.size, "\nnumber of instances", len(df.columns))


# In[57]:


df.describe(include="all")


# In[58]:


df.info()


# In[59]:


df[df["Albumin_and_Globulin_Ratio"].isnull()]


# In[60]:


df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].median(), inplace=True)


# In[61]:


le = preprocessing.LabelEncoder()
le.fit(df.gender.unique())
df['Gender_Encoded'] = le.transform(df.gender)
df.drop(['gender'], axis=1, inplace=True)


# In[62]:


correlations = df.corr()

# and visualize
plt.figure(figsize=(10, 10))
g = sns.heatmap(correlations, cbar = True, square = True, annot=True, fmt= '.2f', annot_kws={'size': 10})


# In[63]:


print(pearsonr(df['Total_Bilirubin'], df['Direct_Bilirubin']))


# In[64]:


px.scatter(df, x='Total_Bilirubin', y='Direct_Bilirubin')


# In[65]:


px.scatter(df, x='Alamine_Aminotransferase', y='Aspartate_Aminotransferase')


# In[66]:


g = sns.PairGrid(df, hue="Dataset", vars=['age','Gender_Encoded','Total_Bilirubin','Total_Protiens'])
g.map(plt.scatter)
plt.show()


# In[67]:


X = df[['age', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens', 
        'Albumin', 'Albumin_and_Globulin_Ratio','Gender_Encoded']]
y = df[['Dataset']]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
print (X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# In[72]:


model = LogisticRegression(penalty='l2', solver='lbfgs', C=0.1)

model.fit(X_train, y_train)
#testing the model
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(X_test.head(4))
print('_'*50)
print("Mean Squared Error (MSE):", mse)

# Perform K-Fold cross-validation
cv = KFold(n_splits=5, shuffle=True, random_state=45)  

scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

print("Cross-Validation Accuracy Scores:")
print(scores)
print('_'*30)
print(f"Mean Cross-Validation Accuracy: {scores.mean():.2f}")

# Evaluating model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print(f"Model Accuracy: {accuracy :.2f}")
print('_'*30)
print("Classification Report:")
print(classification_rep)


# In[71]:


rf = RandomForestClassifier(n_estimators=25, random_state=2018)
rf.fit(X_train, y_train)
rf_predicted = rf.predict(X_test)

random_forest_score      = round(rf.score(X_train, y_train) * 100, 2)
random_forest_score_test = round(rf.score(X_test, y_test) * 100, 2)

print('Random Forest Score: ', random_forest_score)
print('Random Forest Test Score: ', random_forest_score_test)
print('Accuracy: ', accuracy_score(y_test,rf_predicted))
print('\nClassification report: \n', classification_report(y_test,rf_predicted))

g = sns.heatmap(confusion_matrix(y_test,rf_predicted), annot=True, fmt="d")


# In[73]:


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assume X and y are your features and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a logistic regression model with class weights
model = LogisticRegression(class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


# In[74]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Assume X and y are your features and target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest model with class weights
model = RandomForestClassifier(class_weight='balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))


# # Model Performance:
# 
# ## Logistic Regression:
# 
# #### Mean Squared Error (MSE):
# 0.2857142857142857
# 
# #### Cross-Validation Accuracy: 
# 0.72
# 
# #### Test Accuracy: 
# 0.71
# 
# Precision, Recall, F1-score: Higher for class 1 than class 2, indicating an imbalance in class prediction accuracy.
# Interpretability: High. Logistic regression provides coefficients that can be easily interpreted to understand the relationship between features and the target variable.
# 
# 
# ## Random Forest:
# 
# #### Training Score: 100.0 (overfitting on training data)
# 
# #### Test Accuracy: 0.68
# 
# Precision, Recall, F1-score: Similar trends as logistic regression, with higher performance for class 1.
# Interpretability: Moderate. While random forests are less interpretable than logistic regression, feature importances can still provide insights into which features are most influential.
# 
# ## Interpretability:
# 
# ### Logistic Regression: 
# Better for interpretability, as it provides clear insights into feature importance and their relationship with the target variable.
# 
# ### Random Forest: 
# Less interpretable but provides feature importances which can be useful for understanding the data.
# 
# ### Performance:
# 
# Both models have similar overall accuracy, but logistic regression slightly outperforms random forest in terms of cross-validation and test accuracy.

# In[ ]:




