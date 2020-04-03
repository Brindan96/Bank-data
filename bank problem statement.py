# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 11:08:44 2020

@author: msi
"""

# Predicting whether person has subscribed for a term deposit or not

#import the data
bank = pd.read_csv("F:\\Data Science\\Assignemnts\\Brindan\\logistic regression\\bank-full.csv", sep=';')
bank
??pd.read_csv
bank.shape
plt.boxplot(bank)
pd.crosstab(bank.age, bank.y).plot(kind='hist')
sb.countplot(bank.job)
sb.countplot(bank.y) # 39K persons have not suscribed and 5.3K persons have suscribed
sb.boxplot(data=bank, orient='v') 
np.mean(bank)

# Creating dummy values

bank_dummies = pd.get_dummies(bank[["job","marital","education","default","housing","loan","contact","month","poutcome"]])
# Dropping the columns with catagorial data
bank.drop(["job","marital","education","default","housing","loan","contact","month","poutcome"], inplace=True,axis = 1)
#cy=bank.drop(["y"], inplace=True, axis=1)
# combining the dummies and normal data
bank = pd.concat([bank,bank_dummies],axis=1)

bank.shape
sb.countplot(bank.y)

#getting the output variable 
y=bank.iloc[:,[7]]
#dropping the output variable since i want it at the last column
bank.drop(['y'], inplace=True,axis=1)
#combing dropped column with original data
bank=pd.concat([bank,y],axis=1)
#train and test data
X=bank.iloc[:,0:51]
Y=bank.iloc[:,51:52]

# creating the model
model1=LogisticRegression()
#fitting the model 
model1.fit(X,Y)
#coeficients 
model1.coef_
# Predicting whether the person has suscribed or not
pred1=model1.predict(X)
# Combining the predicted values with original data
bank["pred1"] = pred1

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(Y,pred1)
print (confusion_matrix)

# crosstable of actual vs predictions 
pd.crosstab(bank.y, bank.pred1).plot(kind='bar')

from sklearn.metrics import accuracy_score
#to find the accuracy score of model
accuracy_score(Y,pred1) #0.90