import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

#  Voting 
data = pd.read_csv('所有特征.csv',encoding='utf-8',index_col='企业代号')
for i in range(len(data)):
    a='E'+str(i+1)
    if data.loc[a,'是否违约']=='否':
        data.loc[a,'违约']=0
    else :
        data.loc[a,'违约']=1

x = data.iloc[:,:-3].values
y = data.iloc[:,-1].values

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier as ada
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import VotingClassifier
from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import StackingCVClassifier

#算法调参
LR = LogisticRegression(C=0.1, class_weight=None, dual=False, fit_intercept=True,
                        intercept_scaling=1, l1_ratio=None, max_iter=100,
                        multi_class='auto', n_jobs=None, penalty='l2',
                        random_state=None, solver='newton-cg', tol=0.0001, verbose=0,
                        warm_start=False)
Ada = ada(algorithm='SAMME',learning_rate=0.1,
          n_estimators=100, random_state=30)
GBDT = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse', init=None,
                                  learning_rate=0.7, loss='exponential', max_depth=3,
                                  max_features='auto',
                                  max_leaf_nodes=None,
                                  min_impurity_decrease=0.0,
                                  min_samples_leaf=1, min_samples_split=2,
                                  min_weight_fraction_leaf=0.0, n_estimators=25,
                                  n_iter_no_change=None,
                                  random_state=30, subsample=1.0, tol=0.0001,
                                  validation_fraction=0.1, verbose=0,
                                  warm_start=False)
svc = SVC(C=0.8, break_ties=False, cache_size=200, class_weight=None, coef0=0.0,
          decision_function_shape='ovr', degree=3, gamma=20, kernel='rbf',
          max_iter=-1, probability=True, random_state=None, shrinking=True, tol=0.001,
          verbose=False)

rf = RF(bootstrap=True, ccp_alpha=0.0, class_weight=None,
        criterion='gini', max_depth=None, max_features='auto',
        max_leaf_nodes=None, max_samples=None,
        min_impurity_decrease=0.0,
        min_samples_leaf=1, min_samples_split=2,
        min_weight_fraction_leaf=0.0, n_estimators=100,
        n_jobs=None, oob_score=False, random_state=30, verbose=0,
        warm_start=False)


sclf = StackingCVClassifier(classifiers=[Ada, GBDT, LR,rf],
                            use_probas=True,
                            meta_classifier=svc,
                            random_state=30)

#划分训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=30)

#特征缩放
tranfer = StandardScaler() #标准化
x = tranfer.fit_transform(x) #拟合转换
x_train = tranfer.transform(x_train)
x_test = tranfer.transform(x_test)


weight = []
d=[]
#训练模型并评价
for clf, label in zip([LR, Ada, GBDT, svc, rf, sclf],
                      ['LR',
                       'Ada',
                       'GBDT',
                       'svc',
                       'rf',
                       'StackingClassifier']):

    clf.fit(x_train, y_train)#训练集拟合
    y_predict = clf.predict(x_test) #测试集预测值
    print('{}在预测集模型的准确率为：\n'.format(label), metrics.accuracy_score(y_test, y_predict))
    print('{}在训练集模型的准确率为：\n'.format(label), metrics.accuracy_score(y_train, clf.predict(x_train)))
    print('{}的综合准确率为：\n'.format(label), metrics.accuracy_score(y, clf.predict(x)))
    tem = metrics.accuracy_score(y, clf.predict(x))#准确率
    d.append(estimator.predict_proba(x))
    print('{}的ROC面积为：'.format(label), metrics.roc_auc_score(y, clf.predict(x)))



weight
del weight[-1]
# 软投票
w = weight/sum(weight)
vote2= VotingClassifier(estimators=[('LR',LR),('Ada',Ada), ('GBDT',GBDT), ('SVC',svc),('rf',rf)],
                        voting='soft',weights=weight)
vote2.fit(x_train,y_train)
y_predict = vote2.predict(x_test)
print('{}在预测集模型的准确率为：\n'.format('soft Voting'),metrics.accuracy_score(y_test,y_predict))
print('{}在训练集模型的准确率为：\n'.format('soft Voting'),metrics.accuracy_score(y_train,vote2.predict(x_train)))
print('soft voting的综合表现:\n',metrics.accuracy_score(y,vote2.predict(x)))
print('soft voting的ROC面积：\n',roc_auc_score(y,vote2.predict(x)))

# 输出结果如下：
# LR在预测集模型的准确率为：
#  0.9354838709677419
# LR在训练集模型的准确率为：
#  0.9239130434782609
# LR的综合准确率为：
#  0.926829268292683
# LR的ROC面积为： 0.8466435185185186
# Ada在预测集模型的准确率为：
#  1.0
# Ada在训练集模型的准确率为：
#  0.967391304347826
# Ada的综合准确率为：
#  0.975609756097561
# Ada的ROC面积为： 0.9444444444444444
# GBDT在预测集模型的准确率为：
#  0.9354838709677419
# GBDT在训练集模型的准确率为：
#  1.0
# GBDT的综合准确率为：
#  0.983739837398374
# GBDT的ROC面积为： 0.9895833333333333
# svc在预测集模型的准确率为：
#  0.8709677419354839
# svc在训练集模型的准确率为：
#  1.0
# svc的综合准确率为：
#  0.967479674796748
# svc的ROC面积为： 0.9259259259259259
# rf在预测集模型的准确率为：
#  1.0
# rf在训练集模型的准确率为：
#  1.0
# rf的综合准确率为：
#  1.0
# rf的ROC面积为： 1.0
# StackingClassifier在预测集模型的准确率为：
#  0.9354838709677419
# StackingClassifier在训练集模型的准确率为：
#  1.0
# StackingClassifier的综合准确率为：
#  0.983739837398374
# StackingClassifier的ROC面积为： 0.9895833333333333
# 4.853658536585366
# soft Voting在预测集模型的准确率为：
#  1.0
# soft Voting在训练集模型的准确率为：
#  1.0
# soft voting的综合表现:
#  1.0
# soft voting的ROC面积：
#  1.0
# P = vote2.predict_proba(x)[:,1]
# df = pd.DataFrame(data={'违约概率':P})
# P = vote2.predict_proba(x)[:,1]
# df['信誉评级'] = data['信誉评级'].reset_index()['信誉评级']
# fpr,tpr,threshold = metrics.roc_curve(y,P)
# # 计算AUC的值
# roc_auc = metrics.auc(fpr,tpr)
#
# #绘制面积图
# plt.figure(figsize=(6,4),dpi=250)
# plt.stackplot(fpr,tpr,color='steelblue',alpha=0.5,edgecolor='black')
# # 添加边际线
# plt.plot(fpr,tpr,color='black',lw=1)
# # 添加对角线
# plt.plot([0,1],[0,1],color='red',linestyle='--')
# # 添加文本信息
# plt.text(0.5,0.3,'ROC curve (area = %0.4f)' % roc_auc,fontsize=10)
# # 添加x轴坐标与y轴坐标
# plt.xlabel('1-Specificity')
# plt.ylabel('Sensitivity')
# plt.show()
# plt.figure(figsize=(6,4),dpi=250)
# plt.stackplot(fpr,tpr,color='steelblue',alpha=0.5,edgecolor='black')
# # 添加边际线
# plt.plot(fpr,tpr,color='black',lw=1)
# # 添加对角线
# plt.plot([0,1],[0,1],color='red',linestyle='--')
# # 添加文本信息
# plt.text(0.5,0.3,'ROC curve (area = %0.4f)' % roc_auc,fontsize=10)
# # 添加x轴坐标与y轴坐标
# plt.xlabel('1-Specificity')
# plt.ylabel('Sensitivity')
# plt.show()
P = vote2.predict_proba(x)[:,1]
df = pd.DataFrame(data={'违约概率':P})
df['信誉评级'] = data['信誉评级'].reset_index()['信誉评级']
aver_A =0
A_aver = sum(df[df['信誉评级']=='A']['违约概率'])/len(df[df['信誉评级']=='A']['违约概率'])
B_aver = sum(df[df['信誉评级']=='B']['违约概率'])/len(df[df['信誉评级']=='B']['违约概率'])
C_aver = sum(df[df['信誉评级']=='C']['违约概率'])/len(df[df['信誉评级']=='C']['违约概率'])
D_aver = sum(df[df['信誉评级']=='D']['违约概率'])/len(df[df['信誉评级']=='D']['违约概率'])
print(A_aver)
print(B_aver)
print(C_aver)
print(D_aver)
da = {'A企业违约风险':A_aver,'B企业违约风险':B_aver,'C企业违约风险':C_aver,'D企业违约风险':D_aver}
da1 = pd.DataFrame(data=da,index=[0])
da1.to_csv('平均违约风险最终结果.csv',encoding='gbk')