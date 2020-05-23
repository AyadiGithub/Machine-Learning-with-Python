import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import preprocessing

pd.set_option('display.max_columns', None)
df = pd.read_csv(r'loan_train.csv')
df.head()
df.shape

df['due_date'] = pd.to_datetime(df['due_date'])
df['effective_date'] = pd.to_datetime(df['effective_date'])
df.head()
df['loan_status'].value_counts()
df.head()


import seaborn as sns
bins = np.linspace(df.Principal.min(), df.Principal.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'Principal', bins=bins, ec="k")

g.axes[-1].legend()
plt.show()


# Feature Selection/Extraction

df['dayofweek'] = df['effective_date'].dt.dayofweek
bins = np.linspace(df.dayofweek.min(), df.dayofweek.max(), 10)
g = sns.FacetGrid(df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g.map(plt.hist, 'dayofweek', bins=bins, ec="k")
g.axes[-1].legend()
plt.show()


df['weekend'] = df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
df.head()

# Convert Categorical features to numerical values

df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
df.head()

# One Hot Encoding

df.groupby(['education'])['loan_status'].value_counts(normalize=True)

df[['Principal','terms','age','Gender','education']].head()

Feature = df[['Principal','terms','age','Gender','weekend']]
Feature = pd.concat([Feature,pd.get_dummies(df['education'])], axis=1)
Feature.drop(['Master or Above'], axis = 1,inplace=True)
Feature.head()

X = Feature
X[0:5]

y = df['loan_status'].values
y[0:5]

# Normalizing Data

X= preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]


# Classification

#K Nearest Neighbor (KNN)

#Importing train/test split from sklearn and KNN Classifier
#Importing metrics from sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

#Setting up train/test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=6)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Finding the best K Value
Ks = 20
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfusionMx = [];
for n in range(1,Ks):
    
    #Train Model and Predict  
    KNN = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=KNN.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 


k = 5
KNN = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
KNN

yhat = KNN.predict(X_test)

from sklearn.metrics import jaccard_similarity_score
from sklearn.metrics import f1_score

print("Jaccard Score = %.4f" % jaccard_similarity_score(y_test, yhat))
print("F1 Score = %.4f" % f1_score(y_test, yhat, average='weighted') )



# DECISION TREE METHOD
#Importing Decision Tree Classifier from sklearn
from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
DecisionTree.fit(X_train,y_train)

PredictionTree = DecisionTree.predict(X_test)
print (PredictionTree [0:10])
print (y_test [0:10])

print("Jaccard Score = %.4f" % jaccard_similarity_score(y_test, PredictionTree))
print("F1 Score = %.4f" % f1_score(y_test, PredictionTree, average='weighted') )

#Importing Graphviz and Pydotplus
import pydotplus
import graphviz 
from sklearn import tree

Feature.columns[0:8]
filename = "loan_status.png"
featureNames = Feature.columns[0:8]
dot_data=tree.export_graphviz(DecisionTree,feature_names=featureNames, out_file=None, class_names= 'loan_status', filled=True,  special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
gvz_graph = graphviz.Source(graph.to_string())
gvz_graph

# USING SUPPORT VECTOR MACHINE 

#Importing SVM from sklearn
from sklearn import svm
svmloan = svm.SVC(kernel='rbf')
svmloan.fit(X_train, y_train)

y2hat = svmloan.predict(X_test)
y2hat [0:5]


print("Avg F1-score for RBF Kernel: %.4f" % f1_score(y_test, y2hat, average='weighted'))

print("Jaccard score for RBF Kernel: %.4f" % jaccard_similarity_score(y_test, y2hat))

# USING LOGISTIC REGRESSION

#Importing LogisticRegression from sklearn linear model
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
LR
y3hat = LR.predict(X_test)
y3hat_prob = LR.predict_proba(X_test)


#Importing f1_score, Jaccard score and log_loss from sklearn metrics
from sklearn.metrics import log_loss


print("Jaccard Score = %.4f" % jaccard_similarity_score(y_test, y3hat))
print("F1 Score = %.4f" % f1_score(y_test, y3hat, average='weighted') )
print("Log_loss = %.4f" % log_loss(y_test, y3hat_prob))

# Importing the TESTING CSV

test_df = pd.read_csv(r'loan_test.csv')
test_df.head()
test_df.shape
pd.set_option('display.max_columns', None)

# Test Data Preprocessing
test_df['due_date'] = pd.to_datetime(test_df['due_date'])
test_df['effective_date'] = pd.to_datetime(test_df['effective_date'])
test_df.head()
test_df['loan_status'].value_counts()
test_df.head()

import seaborn as sns
bins1 = np.linspace(test_df.Principal.min(), test_df.Principal.max(), 10)
g1 = sns.FacetGrid(test_df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g1.map(plt.hist, 'Principal', bins=bins1, ec="k")
g1.axes[-1].legend()
plt.show()

# Feature Selection/Extraction

test_df['dayofweek'] = test_df['effective_date'].dt.dayofweek
bins1 = np.linspace(test_df.dayofweek.min(), test_df.dayofweek.max(), 10)
g1 = sns.FacetGrid(test_df, col="Gender", hue="loan_status", palette="Set1", col_wrap=2)
g1.map(plt.hist, 'dayofweek', bins=bins1, ec="k")
g1.axes[-1].legend()
plt.show()


test_df['weekend'] = test_df['dayofweek'].apply(lambda x: 1 if (x>3)  else 0)
test_df.head()

# Convert Categorical features to numerical values

test_df.groupby(['Gender'])['loan_status'].value_counts(normalize=True)

test_df['Gender'].replace(to_replace=['male','female'], value=[0,1],inplace=True)
test_df.head()

# One Hot Encoding

test_df.groupby(['education'])['loan_status'].value_counts(normalize=True)

test_df[['Principal','terms','age','Gender','education']].head()

Feature1 = test_df[['Principal','terms','age','Gender','weekend']]
Feature1 = pd.concat([Feature1,pd.get_dummies(test_df['education'])], axis=1)
Feature1.drop(['Master or Above'], axis = 1,inplace=True)
Feature1.head()

XTest = Feature1
XTest[0:5]


yTest = test_df['loan_status'].values
yTest[0:5]


# Normalizing Data

XTest= preprocessing.StandardScaler().fit(XTest).transform(XTest)
XTest[0:5]



# Model Evaluation Using Test Data

# Testing KNN Classifier on Test Data
KNN_predict = KNN.predict(XTest)
print('KNN Jaccard_score = %.4f' % jaccard_similarity_score(yTest, KNN_predict))
print('KNN F1_score = %.4f '% f1_score(yTest, KNN_predict, average='weighted'))
      
# Testing DecisionTree Classifier on Test Data      

PredictionTreeTest = DecisionTree.predict(XTest)

print('DecisionTree Jaccard_score = %.4f' % jaccard_similarity_score(yTest, PredictionTreeTest))
print('DecisionTree F1_score = %.4f' % f1_score(yTest, PredictionTreeTest, average='weighted'))

# Testing Support Vector Machine Classifier on Test Data   

SVMTest = svmloan.predict(XTest)

print('SVM Jaccard_score  = %.4f' % jaccard_similarity_score(yTest, SVMTest))
print('SVM F1_score = %.4f' % f1_score(yTest, SVMTest, average='weighted'))

# Testing Logistic Regression Classifier on Test Data        

LRTest = LR.predict(XTest)
LRTest_prob = LR.predict_proba(XTest)

print('Logistic Regression Jaccard_score = %.4f' % jaccard_similarity_score(yTest, LRTest))
print('Logistic Regression F1_score = %.4f' % f1_score(yTest, LRTest, average='weighted'))
print("Logistic Regression Log_loss = %.4f" % log_loss(yTest, LRTest_prob))