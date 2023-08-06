import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

data = pd.read_csv('所有特征.csv',encoding='UTF-8',index_col='企业代号')
data

for i in range(len(data)):
    a='E'+str(i+1)
    if data.loc[a,'是否违约']=='否':
        data.loc[a,'违约']=0
    else :
        data.loc[a,'违约']=1

x = data.iloc[:,:-3].values
y = data.iloc[:,-1].values
data.iloc[:,:-3].values

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30)
tranfer = StandardScaler()
x=tranfer.fit_transform(x)
x_train = tranfer.fit_transform(x_train)
x_test = tranfer.transform(x_test)


# 预估器流程
from sklearn.model_selection import GridSearchCV
estimator = LogisticRegression()

param_dict={'C':[i*0.1  for i in range(1,11)],
            'penalty':['l1', 'l2','elasticnet'],
            'solver':['newton-cg','lbfgs','liblinear','sag','saga'],
            'max_iter':[100,200,300]     }

estimator = GridSearchCV(estimator,param_grid=param_dict,cv=5,scoring='accuracy')

estimator.fit(x_train,y_train)

from sklearn import metrics
y_predict = estimator.predict(x_test)


#LR
print('最佳参数：\n',estimator.best_params_)
print('最佳结果：\n',estimator.best_score_)
print('最佳估计器：\n',estimator.best_estimator_)
print('模型评估报告：\n',metrics.classification_report(y_test,y_predict))

print('预测集模型的准确率为：\n',metrics.accuracy_score(y_test,y_predict))
print('训练集模型的准确率为：\n',metrics.accuracy_score(y_train,estimator.predict(x_train)))

d1 = estimator.predict_proba(x)

#ada

from sklearn.ensemble import AdaBoostClassifier as ada
estimator = ada(algorithm='SAMME', base_estimator=None, learning_rate=0.1,
                   n_estimators=100, random_state=30)

#param_dict={'n_estimators':[100,200,300],
     #       'learning_rate':[i*0.1 for i in range(1,11)],
      #      'algorithm':['SAMME','SAMME.R']}

#estimator = GridSearchCV(estimator,param_grid=param_dict,cv=5,scoring='accuracy')

estimator.fit(x_train,y_train)
y_predict = estimator.predict(x_test)


#
# print('最佳参数：\n',estimator.best_params_(y_predict))
# # print('最佳结果：\n',estimator.best_score_())
# # print('最佳估计器：\n',estimator.best_estimator_)
print('模型评估报告：\n',metrics.classification_report(y_test,y_predict))
print('预测集模型的准确率为：\n',metrics.accuracy_score(y_test,y_predict))
print('训练集模型的准确率为：\n',metrics.accuracy_score(y_train,estimator.predict(x_train)))
d2 = estimator.predict_proba(x)

#GBDT
from sklearn.ensemble import GradientBoostingClassifier as GBDT

estimator = GBDT(random_state=30)

param_dict={'loss':['exponential'],
            'learning_rate':[0.7],
            'n_estimators':[25],
            'max_features':['auto'],
            'max_depth':[3],
            'min_samples_split':[2]
           }

estimator = GridSearchCV(estimator,param_grid=param_dict,cv=5,scoring='accuracy')

estimator.fit(x_train,y_train)

y_predict = estimator.predict(x_test)



print('最佳参数：\n',estimator.best_params_)
print('最佳结果：\n',estimator.best_score_)
print('最佳估计器：\n',estimator.best_estimator_)
print('模型评估报告：\n',metrics.classification_report(y_test,y_predict))

print('预测集模型的准确率为：\n',metrics.accuracy_score(y_test,y_predict))
print('训练集模型的准确率为：\n',metrics.accuracy_score(y_train,estimator.predict(x_train)))
estimator.predict(x)

d3 =estimator.predict_proba(x)

#SVC
from sklearn.svm import SVC

# estimator = SVC(random_state=30)

# param_dict={'kernel':['poly','linear','rbf','sigmoid'],
#             'C':[i*0.1 for i in range(1,11)],
#             'degree':[3,4,5],
#            'gamma':['scale','auto']
#            }

estimator = SVC(C=0.8, cache_size=200, class_weight=None, coef0=0.0,
  decision_function_shape='ovr', degree=3, gamma=20, kernel='rbf',
  max_iter=-1, probability=True, random_state=None, shrinking=True,
  tol=0.001, verbose=False)

estimator.fit(x_train,y_train.ravel())
y_predict = estimator.predict(x_test)




print('模型评估报告：\n',metrics.classification_report(y_test,y_predict))

print('预测集模型的准确率为：\n',metrics.accuracy_score(y_test,y_predict))
print('训练集模型的准确率为：\n',metrics.accuracy_score(y_train,estimator.predict(x_train)))
estimator.predict_proba(x)
y_train.ravel().shape

#randomforest
from sklearn.ensemble import RandomForestClassifier as RF

estimator = RF(random_state=30)

param = {'n_estimators':[100],
        'criterion':['gini'],
         'min_samples_split':[2,4,6,8],
         'max_features':['auto']
        }
estimator = GridSearchCV(estimator=estimator,param_grid=param,cv=5,
                   scoring='accuracy')

estimator.fit(x_train,y_train)
y_predict = estimator.predict(x_test)



print('最佳参数：\n',estimator.best_params_)
print('最佳结果：\n',estimator.best_score_)
print('最佳估计器：\n',estimator.best_estimator_)
print('模型评估报告：\n',metrics.classification_report(y_test,y_predict))

print('预测集模型的准确率为：\n',metrics.accuracy_score(y_test,y_predict))
print('训练集模型的准确率为：\n',metrics.accuracy_score(y_train,estimator.predict(x_train)))
d4 = estimator.predict_proba(x)
d=np.zeros((123,4))
d=pd.DataFrame(d)
d.columns=['逻辑回归','RF','Adaboost','GBDT']
d['逻辑回归']=d1[:,1]
d['Adaboost']=d2[:,1]
d['GBDT']=d3[:,1]
d['RF']=d4[:,1]
d.to_csv('dinal.cvs',encoding='UTF-8')