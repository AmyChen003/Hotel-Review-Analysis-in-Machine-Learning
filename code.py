from sklearn import tree
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
from matplotlib import pyplot as plt
import graphviz
from IPython.display import Image, display
import pydotplus
import matplotlib.image as img
import pydot
from sklearn.metrics import roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from sklearn.model_selection import GridSearchCV
data = pd.read_csv("data.csv")
#print data.columns.values
feature= data.iloc[:,8:15]
feature1= data.iloc[:,8:15]
feature2= data.iloc[:,0:16]
#print feature
label=data.iloc[:,16]
#print label


for i in range(len(label)):
    if (label.loc[label[i]] == 1 or label.loc[label[i]] == 2 or label.loc[label[i]] == 3):
        label.at[i]=0
        # label.loc[label[i]]= 0
    else:
        label.at[i] = 1
# print label


lb = LabelEncoder()
for col in feature.columns:
    feature[col] = lb.fit_transform(feature[col])
#print feature
# feature1=lb.inverse_transform(feature)
# print feature1
#
x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size = 0.2)

clf = tree.DecisionTreeClassifier(criterion='gini')
clf.fit(x_train, y_train)
# print "clf:"+str(clf)

# dot_data = tree.export_graphviz(clf, out_file=None,feature_names=feature.columns)
#
# graph = graphviz.Source(dot_data)
# graph.render("hotels")

with open("hotels.dot", 'w') as f:
    f = tree.export_graphviz(clf,
                            feature_names=feature1.columns.values,
                            filled=True, rounded=True,
                            out_file=f)
(graph,) = pydot.graph_from_dot_file("hotels.dot")
graph.write_png("hotels.png")

plt.imshow(img.imread('hotels.png'))
plt.show()
#
#
y_pred = clf.predict(x_test)
y_test=y_test.values
print "Accuracy 1 is ",accuracy_score(y_test, y_pred)


feature=feature.values
label=label.values

kf1 = KFold(n_splits=10,shuffle = True)
for train_index, test_index in kf1.split(feature):
    X_train, X_test = feature[train_index], feature[test_index]
    Y_train, Y_test = label[train_index], label[test_index]
    clf = tree.DecisionTreeClassifier(criterion='gini')
    clf.fit(X_train, Y_train)
    y_pred=clf.predict(X_test)
    print "Accuracy is ", accuracy_score(Y_test, y_pred)

# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, predictedY)
# roc_auc = auc(false_positive_rate, true_positive_rate)
# plt.figure(figsize=(10,10))
# plt.title('Receiver Operating Characteristic')
# plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],linestyle='--')
# plt.axis('tight')
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(feature, label, test_size=0.25)
clf = LogisticRegression()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
acc=accuracy_score(y_test, y_pred)
print "Accuracy 2 is ", acc
# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, pred)
# roc_auc = auc(false_positive_rate, true_positive_rate)
# plt.figure(figsize=(6,5))
# plt.title('Logistic Regression-ROC')
# plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],linestyle='--')
# plt.axis('tight')
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()

LR_model= LogisticRegression()

tuned_parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] ,
              'penalty':['l1','l2']
                   }
LR= GridSearchCV(LR_model, tuned_parameters,cv=10)
LR.fit(x_train,y_train)
y_prob = LR.predict_proba(x_test)[:,1]
y_pred = np.where(y_prob > 0.5, 1, 0)
LR.score(x_test, y_pred)
acc=accuracy_score(y_test, y_pred)
print "Accuracy 3 is ", acc
# false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_prob)
# roc_auc = auc(false_positive_rate, true_positive_rate)
# plt.figure(figsize=(6,5))
# plt.title('GridSearch-ROC')
# plt.plot(false_positive_rate,true_positive_rate, color='red',label = 'AUC = %0.2f' % roc_auc)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],linestyle='--')
# plt.axis('tight')
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.show()
