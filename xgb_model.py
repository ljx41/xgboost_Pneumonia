import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from multimodel import multi_models_roc
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import shap
import os
from sklearn.metrics import accuracy_score, classification_report
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['font.family'] = ['sans-serif']


df1 = pd.read_excel('process_data.xlsx')
df2 = df1.drop('apachescore',axis=1)
df2 = df2.drop('verbal',axis=1)
df2 = df2.drop('eyes',axis=1)
df2 = df2.drop('motor',axis=1)
df2 = df2.drop('admissionheight',axis=1)
df2 = df2.drop('admissionweight',axis=1)
data_x = df2.iloc[:,:-1]
data_y = df2.iloc[:,-1]

for item in data_x.columns[1:]:
    mean_val = data_x[item].mean()
    data_x[item].fillna(mean_val,inplace=True)
    print(item+'###########complete###############')

data_x.iloc[:,1:]= (data_x.iloc[:,1:]-data_x.iloc[:,1:].min())/(data_x.iloc[:,1:].max()-data_x.iloc[:,1:].min())

X_train, X_test, y_train,  y_test = train_test_split(data_x, data_y.values.ravel(), test_size=0.15,random_state=41)

plt.figure(figsize=(20, 5))
gs = GridSpec(1, 2)

uniq, counts = np.unique(y_train, return_counts=True)
ax1 = plt.subplot(gs[0, 0])

pd.Series(dict(zip(uniq, counts))).plot(kind='bar', title="Orginal Class Distribution", ax=ax1, color=['aqua', 'darkorange'])
plt.xlabel("Pneumonia death, Non death group")
plt.ylabel("Number of Samples")

## data over-sampling
X_samping, y_sampling = ADASYN(random_state=0).fit_resample(X_train.iloc[:,1:], y_train)
uniq, counts = np.unique(y_sampling, return_counts=True)
ax2 = plt.subplot(gs[0, 1])
                  
pd.Series(dict(zip(uniq, counts))).plot(kind='bar', title="Class Distribution After ADASYN", ax=ax2, color=['aqua', 'darkorange'])
plt.xlabel("Pneumonia death, Non death group")
plt.ylabel("Number of Samples")
plt.savefig(os.path.join('Figure', "figure_sampling"))
plt.close()




xgb_model = xgb.XGBClassifier(max_depth=20,learning_rate=0.0005,n_estimators=250)
xgb_model.fit(X_samping, y_sampling)#训练模型
importance = xgb_model.get_booster().get_score()
tuples = [(k, importance[k]) for k in importance]
tuples = sorted(tuples, key=lambda x: x[1],reverse=True)
feature_names,scores = map(list,zip(*tuples))#zip(*)是对元组解压缩
df = pd.DataFrame({'feature_names':feature_names,'scores':scores})
df.to_csv('筛选特征.csv',index=None)


#基于特征筛选结果，建立模型
item_list = pd.read_csv('筛选特征.csv')['feature_names']

X_train_selection = X_samping[item_list]
X_test_selection = X_test[item_list]

#xgb model
xgb_model = xgb.XGBClassifier(max_depth=41,learning_rate=0.0007,n_estimators=200)
xgb_model.fit(X_train_selection,y_sampling)
#svm model
svm_model = SVC(probability=True) 
svm_model.fit(X_train_selection,y_sampling)
#lr model
lr_model = LogisticRegression()
lr_model.fit(X_train_selection,y_sampling)
#rf_model 
rf_model = RandomForestClassifier()
rf_model.fit(X_train_selection,y_sampling)

y_pred = xgb_model.predict(X_test_selection)
print(classification_report(y_pred, y_test))
print()
print("The final accuracy is %.2f" % accuracy_score(y_pred, y_test))


from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

for model_temp in (xgb_model,svm_model,lr_model,rf_model):
    print("模型间隔")
    y_pred = model_temp.predict(X_test_selection)
    print(average_precision_score(y_pred,y_test))
    print(f1_score(y_pred,y_test))
    print(accuracy_score(y_pred,y_test))

y_pred = lr_model.predict(X_test_selection)
print(average_precision_score(y_pred,y_test))
print(f1_score(y_pred,y_test))
print(accuracy_score(y_pred,y_test))





# evf = evaluation(xgb,X_train,y_train,X_test,y_test,'xgb',2)
# evf2 = evaluation(svm_model,X_train,y_train,X_test,y_test,'svm',2)

names = ['logistic regression','svm','xgboost','random forest']
multi_models = [xgb_model,svm_model,lr_model,rf_model]
colors = ['green','b',
    'crimson',
          'orange'  ]
plt = multi_models_roc(names,multi_models,colors,X_test_selection,y_test,save='False')
# plt.savefig(os.path.join('Figure', "ROC曲线"))
plt.close()

X_shap = X_train[item_list]
explainer = shap.Explainer(xgb_model,X_shap)
shap_values = explainer(X_shap)

# shap.plots.force(shap_values)


shap.plots.bar(shap_values,max_display=20,show_data=False,show=False)
plt.show()
shap.plots.waterfall(shap_values,max_display=20)

shap.plots.beeswarm(shap_values,max_display=20)


shap.plots.force(shap_values[16],figsize=(20, 3),matplotlib=True,contribution_threshold=0.02,text_rotation=45)


 