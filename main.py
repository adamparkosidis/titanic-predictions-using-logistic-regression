import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


########################
### Explore The Data ###
########################

path = os.getcwd()
train = pd.read_csv(path+'/data/train.csv')
train.info()

train.head()

train.isnull() # check for missing data
plt.figure(figsize=(20,16))
sns.heatmap(train.isnull(), yticklabels=False, cbar= False, cmap='viridis')  # wemiss some info in 'Age', 'Embarked' and a lot more in the 'Cabin' column.
plt.show()

# seperate data into numerical and categorical

df_num= train[['Age','SibSp','Parch','Fare']]
df_str = train[['Survived','Pclass','Sex','Ticket','Cabin','Embarked']]

# look for correlations in the numerical data 

print(df_num.corr())
sns.heatmap(df_num.corr(),annot=True)
plt.show()

# compare survival rate across Age, SibSp, Parch, and Fare 

pd.pivot_table(train, index = 'Survived', values = ['Age','SibSp','Parch','Fare'])

# lets check the number of survivors and non survivors

sns.set_style('whitegrid')
print(pd.pivot_table(train, index = 'Survived', values = 'Ticket' ,aggfunc ='count'))
sns.countplot(data=train, x='Survived') 
plt.show()

# lets check the number of male and female survivors and non survivors respectively 

print(pd.pivot_table(train, index = 'Survived', columns = 'Sex', values = 'Ticket' ,aggfunc ='count'))
sns.countplot(data=train, x='Survived',hue = 'Sex')
plt.show()

# lets check the number of survivors and non survivors based on their  Ticket Class 

print(pd.pivot_table(train, index = 'Survived', columns = 'Pclass', values = 'Ticket' ,aggfunc ='count'))
sns.countplot(data=train, x='Survived',hue = 'Pclass') 
plt.show()

# lets check the number of survivors and non survivors based on their  Embarked place

print(pd.pivot_table(train, index = 'Survived', columns = 'Embarked', values = 'Ticket' ,aggfunc ='count'))
sns.countplot(data=train, x='Survived',hue = 'Embarked') 
plt.show()

# lets check the Age of the passengers for whom we have info
sns.displot(data=train.dropna(), x='Age')
plt.show()

# lets check the survival rate of different age groups

labels = ['Age_group_1','Age_group_2','Age_group_3','Age_group_4','Age_group_5','Age_group_6', 'Age_group_7','Age_group_8']
train['Age_group']=pd.cut(train['Age'], bins=8,labels=labels, right=False)
print(pd.pivot_table(train, index = 'Survived', columns = 'Age_group', values = 'Ticket' ,aggfunc ='count'))

plt.figure(figsize=(20,16))
ax=sns.countplot(data=train, x='Age_group',hue = 'Survived')
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

train.drop('Age_group', axis=1, inplace=True)

# lets check the number of passengers that traveled alone or with other people (siblings etc)
sns.countplot(data=train, x='SibSp')
plt.show()

SurvBySibSp=train.groupby([train['Survived'],train['SibSp']>0]).count()
print(SurvBySibSp)

SurvBySibSp.unstack()['PassengerId'].plot(kind='bar',figsize=(12,8))
plt.show()

print(pd.pivot_table(train, index = 'Survived', columns = 'SibSp', values = 'Ticket' ,aggfunc ='count'))
sns.countplot(data=train, x='Survived', hue='SibSp')
plt.legend(loc='upper right')
plt.show()

# lets check the number of passengers that traveled alone or with other people (parents etc)
sns.countplot(data=train, x='Parch')
plt.show()

SurvByParch=train.groupby([train['Survived'],train['Parch']>0]).count()
print(SurvByParch)

plt.figure(figsize=(20,16))
sns.countplot(data=train, x='Survived', hue='Parch')
plt.show()
print(pd.pivot_table(train, index = 'Survived', columns = 'Parch', values = 'Ticket' ,aggfunc ='count'))

# lets check how much people paid

print(train['Fare'].value_counts())
sns.displot(data=train, x='Fare', bins=30)
plt.show()

# let see what people paid (avarage) based on the place they embarked the ship
plt.figure(figsize=(10,7))
sns.boxplot(x='Embarked', y='Fare', data=train)
plt.show()

sns.jointplot(data=train,x='Fare',y='Age',hue='Embarked')
plt.show()

sns.jointplot(data=train,x='Pclass',y='Fare',hue='Survived')
plt.show()

sns.jointplot(data=train,x='Age',y='Fare',hue='Survived')
plt.show()

# Some tickets have only numbers, while other tickets contain also letters. Lets try to find motifs based on theat difference

train['Numeric_Tickets'] = train['Ticket'].apply(lambda x: True if x[0].isnumeric()==True else False)

# lets check the number of survivors and non survivors

print(pd.pivot_table(train, index = 'Survived', columns = 'Numeric_Tickets', values = 'Ticket' ,aggfunc ='count'))
sns.countplot(data=train, x='Survived',hue = 'Numeric_Tickets') 

# Maybe the first letter/number of the Ticket can give us some info

first_num = []
first_letter = []
for i in train['Ticket'].values:
    if i[0].isnumeric(): 
        first_num.append(i[0])
    else: 
        first_letter.append(i[0])

print(np.unique(first_num,return_counts=True))
print(np.unique(first_letter,return_counts=True))

train['Ticket_first_char'] = train['Ticket'].apply(lambda x: x[0])

numeric = train[train['Ticket'].apply(lambda x: x[0].isnumeric())]
print(pd.pivot_table(numeric, index = 'Survived', columns = 'Ticket_first_char', values = 'Ticket' ,aggfunc ='count'))
plt.figure(figsize=(20,16))
sns.countplot(data=numeric, x='Survived',hue = 'Ticket_first_char',palette='colorblind') 
plt.show()

letter = train[train['Ticket'].apply(lambda x: x[0].isnumeric())== False]
print(pd.pivot_table(letter, index = 'Survived', columns = 'Ticket_first_char', values = 'Ticket' ,aggfunc ='count'))
plt.figure(figsize=(20,16))
sns.countplot(data=letter, x='Survived',hue = 'Ticket_first_char',palette='colorblind') 
plt.show()


######################
### Clean The Data ###
######################

# We miss some data associated with the Age of some passengers, we will try to impute that missing data based on a reasonable guess

plt.figure(figsize=(10,7))
sns.boxplot(x='Pclass', y='Age', data=train)
plt.show()


def impute_age(cols):
    '''
    It seems like a good approach to impute the Age based on the passenger class. For example older people may had more time to accumulate wealth and 
    be in the first class etc.
    '''
    Age = cols[0]
    Pclass = cols[1]
    if pd.isnull(Age):
        if Pclass == 1:
            return 38
        elif Pclass == 2:
            return 30
        else:
            return 25
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(impute_age, axis=1)
sns.heatmap(train.isnull(), yticklabels=False, cbar= False, cmap='viridis') # filled 'Age' column with reasonable guesses
plt.show()

# We miss some data associated with the Embarked place of some passengers, we will try to impute that missing data based on a reasonable guess 

train[train['Embarked'].isnull()]
train[(train['Pclass']==1) & (train['Sex']=='female') & (train['Survived']==1)]['Embarked'].value_counts()
# Based on the Pclass and the Embarked place we impute 'Fare' with reasonable guesses
train['Embarked'].iloc[61] = 'C'  # Initially we marked the Ebarked place as 'S', but given that 'C' is also very probable and that the model perform slighlty better we end u[ with 'C'
train['Embarked'].iloc[829] = 'C'

# We decide to drop the Cabin column due to the fact that we miss the biggest part of the data 
train.drop('Cabin', axis=1, inplace=True)

# It is interesting to to use the Name column and try to find some info eg based on the titles prefixing a person's name

train["Salutation"] = train['Name'].apply(lambda x: x.strip().split('.'))
train["Salutation"] = train["Salutation"].apply(lambda x: x[0].split(','))
train["Salutation"] = train["Salutation"].apply(lambda x: x[1])
print(train["Salutation"].value_counts())
train["Salutation"].hist(figsize=(20,7),bins=50)  

train=train[train["Salutation"].apply(lambda x: x in [' Col', ' Dr', ' Major',' Master', ' Miss',  ' Mr', ' Mrs', ' Ms', ' Rev'])]
print(np.unique(train["Salutation"], return_counts=True))

print(pd.pivot_table(train, index = 'Survived', columns = 'Salutation', values = 'Ticket' ,aggfunc ='count'))
plt.figure(figsize=(20,16))
sns.countplot(data=train, x='Survived', hue='Salutation')
plt.show()

'''
We can clearly see some patterns here eg more people with the 'Mrs' and 'Miss' prefixs survived, while more people with the prefix 'Mr' did not
'''

# create dummy variables

sex=pd.get_dummies(train['Sex'],drop_first=True)
embark=pd.get_dummies(train['Embarked'],drop_first=True)
salutation = pd.get_dummies(train['Salutation'],drop_first=True)

train = pd.concat([train,sex,embark,salutation], axis=1)
train.head()

train.drop(['PassengerId','Name','Sex','Ticket','Numeric_Tickets','Ticket_first_char','Fare','Embarked','Salutation'],axis=1,inplace=True)
train.head()
train.dropna()



########################
### Train Test Split ###
########################

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

X_train, X_test, y_train, y_test = train_test_split(train.drop('Survived',axis=1),train['Survived'], test_size=0.30, random_state=101) 
logmodel = LogisticRegression(max_iter=10000)
logmodel.fit(X_train.values,y_train)
predictions = logmodel.predict(X_test.values)

# Evalute my Model - Logistic Regression

print(classification_report(y_test,predictions))
print('Accuracy {:.6f}'.format(accuracy_score(predictions, y_test)))


# Evalute my Model - Random Forest Classifier

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=300)
rfc.fit(X_train,y_train)
predictions_2 = rfc.predict(X_test)
print(classification_report(y_test,predictions_2))
print('Accuracy {:.6f}'.format(accuracy_score(predictions_2, y_test)))

'''
We see that the logistic regression model ends up with an accuracy around 87.16% outmatching the random forest model
'''