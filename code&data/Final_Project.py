#!/usr/bin/env python
# coding: utf-8

# # Employee Attrition Case - Final Project_Hash Analytics

# ## Exploratory Data Analysis

# ### Import Relevant Libraries

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Load Datasets

# In[2]:


currentEmp = pd.read_excel('employeeAttrition.xlsx', sheet_name='CurrentEmployees')
churnedEmp = pd.read_excel('employeeAttrition.xlsx', sheet_name='ChurnedEmployees')


# In[3]:


churnedEmp


# In[4]:


currentEmp


# ### Concatenate Datasets

# In[5]:


# Create an additional column, 'churned' for both datasets
churnedEmp['churned'] = [1] * churnedEmp.shape[0]
currentEmp['churned'] = [0] * currentEmp.shape[0]


# In[6]:


dataset= currentEmp.append(churnedEmp, ignore_index=True, sort=False)
dataset


# In[ ]:





# In[7]:


dataset['Work_accident'] = dataset['Work_accident'].replace({0:'No',1:'Yes'}) 
dataset['promotion_last_5years'] = dataset['promotion_last_5years'].replace({0:'No',1:'Yes'}) 


# In[8]:


dataset.dtypes


# In[9]:


dataset.shape


# In[10]:


dataset.isnull().sum()


# ### Univariate Analysis

# In[11]:


num_cols = ['satisfaction_level', 'last_evaluation',
       'average_montly_hours']

cat_cols = ['Work_accident','time_spend_company','number_project','promotion_last_5years','dept', 'salary', 'churned']


# In[12]:


rowCnt = 4
colCnt = 2     # cols:  overall, no disease, disease
subCnt = 1     # initialize plot number

fig = plt.figure(figsize=(20,30))

for i in cat_cols:
    # OVERALL subplots
    fig.add_subplot(rowCnt, colCnt, subCnt)
    plt.subplots_adjust(hspace=0.6, wspace=0.2)
    plt.xlabel(i, fontsize=20)
    plt.ylabel('Count', fontsize=20)
    plt.xticks(fontsize=20, rotation=90)
    plt.yticks(fontsize=20)

    sns.countplot(dataset[i])
    subCnt = subCnt + 1
plt.savefig('countplots.png')


# In[13]:


rowCnt = 2
colCnt = 2     # cols:  overall, no disease, disease
subCnt = 1     # initialize plot number

fig = plt.figure(figsize=(16,15))

for i in num_cols:
    # OVERALL subplots
    fig.add_subplot(rowCnt, colCnt, subCnt)
    plt.subplots_adjust(hspace=0.2, wspace=0.19)
    plt.xlabel(i, fontsize=20)
    plt.xticks(fontsize=20,rotation=90)
    plt.yticks(fontsize=20)
    #plt.hist(dataset[i])
    sns.distplot(dataset[i])
    subCnt = subCnt + 1
plt.savefig('distributionplots.png')


# ### Bivarite Analysis

# In[14]:


fig = plt.figure(figsize=(15,8))
sns.boxplot(x=dataset['time_spend_company'],y=dataset['average_montly_hours'])
plt.xlabel('Number of years spent with Company',fontsize=20)
plt.ylabel('Average Monthly hours',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('A Boxplot showing the variability of average hours of work across years spent in Company',size=20)
plt.savefig('years_avg_Monthly_hours.png')


# In[15]:


fig = plt.figure(figsize=(22,10))
sns.boxplot(x=dataset['dept'],y=dataset['time_spend_company'])
plt.xlabel('Department',fontsize=20)
plt.ylabel('Number of Years in the Company',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('A Boxplot showing the variability of years spent with company across department',size=20)
plt.savefig('years_dept.png')


# In[16]:


fig = plt.figure(figsize=(15,8))
sns.boxplot(x=dataset['promotion_last_5years'],y=dataset['time_spend_company'])
plt.xlabel('Whether Promoted',fontsize=20)
plt.ylabel('Number of Years in the Company',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('A Boxplot showing the variability of years spent with company and whether promotion occured or not',size=20)
plt.savefig('year_promotion.png')


# In[17]:


fig = plt.figure(figsize=(15,8))
sns.boxplot(x=dataset['salary'],y=dataset['time_spend_company'])
plt.xlabel('Salary',fontsize=20)
plt.ylabel('Number of Years in the Company',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('A Boxplot showing the variability of years spent with company across Salary',size=20)
plt.savefig('year_salary.png')


# In[18]:


fig = plt.figure(figsize=(15,8))
sns.boxplot(x=dataset['number_project'],y=dataset['time_spend_company'])
plt.xlabel('Number of Project',fontsize=20)
plt.ylabel('Number of Years in the Company',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('A Boxplot showing the variability of years spent with company across Number of Projects',size=20)
plt.savefig('year_projects.png')


# In[19]:


fig = plt.figure(figsize=(15,8))
sns.boxplot(x=dataset['promotion_last_5years'],y=dataset['satisfaction_level'])
plt.xlabel('Whether Promoted',fontsize=20)
plt.ylabel('Satisfaction level in the Company',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('A Boxplot showing the variability of satisfaction level across Promotion',size=20)
plt.savefig('satisfaction_promotion.png')


# In[20]:


fig = plt.figure(figsize=(15,8))
sns.boxplot(x=dataset['Work_accident'],y=dataset['time_spend_company'])
plt.xlabel('Work Accident',fontsize=20)
plt.ylabel('Years Spent in Company',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('A Boxplot showing the variability of years spent with company and Work Accident',size=20)
plt.savefig('accident_years.png')


# In[21]:


fig = plt.figure(figsize=(15,8))
sns.boxplot(x=dataset['Work_accident'],y=dataset['satisfaction_level'])
plt.xlabel('Work Accident',fontsize=20)
plt.ylabel('Satisfaction Level',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('A Boxplot showing the variability of satisfaction levels among employees with work accident or not',size=20)
plt.savefig('satisfaction_accident.png')


# In[22]:


fig = plt.figure(figsize=(15,8))
sns.boxplot(x=dataset['number_project'],y=dataset['satisfaction_level'])
plt.xlabel('Number of Project',fontsize=20)
plt.ylabel('Satisfaction Level',fontsize=20)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.title('A Boxplot showing the variability of satisfaction levels across number of project',size=20)
plt.savefig('satisfaction_project.png')


# In[23]:


cols = ['satisfaction_level', 'last_evaluation', 'number_project',
       'average_montly_hours', 'time_spend_company']


# In[24]:


fig = plt.figure(figsize=(20,20))
for i in range(len(cols)):
    plt.subplot(3,3,i+1)
    sns.boxplot(x=dataset['churned'],y=dataset[cols[i]])
    plt.subplots_adjust(hspace=0.2, wspace=0.3)
    plt.ylabel(cols[i],fontsize=20)
    plt.xlabel('Churned',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
plt.savefig('churned_others.png')
plt.show()

    
    


# In[25]:


dataset.head()


# In[26]:


data_model = dataset.iloc[:,1:]
data_model


# In[27]:


fig = plt.figure(figsize=(10,8))
sns.heatmap(data_model[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company']].corr(),annot=True);


# ## Preprocessing 

# ### Label Encoding

# In[28]:


from sklearn.preprocessing import LabelEncoder

lb_make = LabelEncoder()
cols = ['Work_accident','promotion_last_5years','dept','salary']
for i in cols:
    
    data_model[i] = lb_make.fit_transform(data_model[i])


# In[29]:


curr_employees = currentEmp.copy().iloc[:,1:]
for i in cols:
    
    curr_employees[i] = lb_make.fit_transform(curr_employees[i])


# In[30]:


curr_employees


# In[ ]:





# ### Standardization

# In[31]:


from sklearn.preprocessing import MinMaxScaler
ml_data = data_model.iloc[:,:-1]
scale = MinMaxScaler()
ml_data = scale.fit_transform(ml_data.iloc[:,:])
cols = ['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','dept','salary']
ml_data = pd.DataFrame(ml_data, columns = cols)
ml_data.head()


# ### Feature Engineering

# In[32]:


#importing the necessary libraries
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.tree import DecisionTreeClassifier# Sequential Forward Selection(sfs)

X = ml_data
y = data_model['churned']
sfs = SFS(DecisionTreeClassifier(),
           k_features=(3,9),
           forward=True,
           floating=False,
           scoring = 'accuracy',
           cv = 0)


# In[33]:


sfs.fit(X, y)
sfs.k_feature_names_ 


# In[34]:


from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()


# In[35]:


sfs1 = SFS(DecisionTreeClassifier(),
           k_features=(3,9),
           forward=True,
           floating=True,
           scoring = 'accuracy',
           cv = 0)

sfs1.fit(X, y)
sfs1.k_feature_names_ 


# In[36]:


from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
fig1 = plot_sfs(sfs1.get_metric_dict(), kind='std_dev')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()


# ## Modelling

# In[37]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[38]:


X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.30, random_state=0)
rf = RandomForestClassifier(n_estimators=20)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
rf_probs = rf.predict_proba(X_test)[:, 1]


# In[39]:


print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print(accuracy_score(y_test, y_pred))


# ### Saving and Loading the model

# In[40]:


import pickle 
  
# Save the trained model as a pickle string. 
saved_model = pickle.dumps(rf) 
  
# Load the pickled model 
rf_from_pickle = pickle.loads(saved_model) 
  


# In[ ]:





# In[41]:


# Use the loaded pickled model to make predictions 
result = rf_from_pickle.predict_proba(curr_employees.iloc[:,:-1]) 


# ### Joining the predicted values to the original DataFrame

# In[42]:


result = [i[1] for i in result]
currentEmp


# In[43]:


currentEmp = currentEmp.drop('churned', axis=1)

currentEmp['likely_to_churn'] = result


# In[44]:


currentEmp.sort_values(by=['likely_to_churn'],ascending=False)


# ## Final Action

# In[45]:


currentEmp['State'] = np.where(currentEmp['likely_to_churn'] >= 0.5, 'Likely to Churn', 'Unlikely to churn')


# In[46]:


fig = plt.figure(figsize=(15,7))
sns.countplot(currentEmp['State'])
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.ylabel('State',fontsize=20)
plt.xlabel('Count', fontsize=20)
plt.title('Proportion of Employees likely to churn',size=20)
plt.savefig('likely_churning.png')
plt.show()


# In[47]:


currentEmp.to_csv('final_prediction.csv', index=False)


# 
