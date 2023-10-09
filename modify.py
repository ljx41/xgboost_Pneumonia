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
import shap
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import StackingClassifier



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

X_train, X_test, y_train,  y_test = train_test_split(data_x, data_y.values.ravel(), test_size=0.1,random_state=41)
X_samping, y_sampling = ADASYN(random_state=0).fit_resample(X_train.iloc[:,1:], y_train)
item_list = pd.read_csv('筛选特征.csv')['feature_names']

X_train_selection = X_samping[item_list]
X_test_selection = X_test[item_list]

#xgb_model = xgb.XGBClassifier(max_depth=41,learning_rate=0.0007,n_estimators=200)

# base_classifiers = [
#     ('xgb', xgb.XGBClassifier(max_depth=2,subsample=0.8,colsample_bytree=0.8,learning_rate=0.001,reg_alpha=0.1,  reg_lambda=0.1,n_estimators=1000,objective='binary:logistic')),
#     ('svm', SVC(probability=True)),
#     ('lr',LogisticRegression()),
#     ('rf', RandomForestClassifier())
# ]
# meta_classifier = LogisticRegression()

# # 创建 stacking 模型
# stacking_model = StackingClassifier(
#     estimators=base_classifiers,
#     final_estimator=meta_classifier
# )



stacking_model = MLPClassifier(
    hidden_layer_sizes=(128,),  # 指定隐藏层的神经元数量，可以是一个元组表示多层结构
    activation='relu',          # 激活函数，可以是 'logistic'、'tanh' 或 'relu'
    solver='adam',              # 优化器，可以是 'adam'、'sgd' 等
    alpha=0.0001,               # 正则化参数
    max_iter=1000,               # 最大迭代次数
    random_state=42
)
stacking_model.fit(X_train_selection, y_sampling)
# # 在测试集上进行预测
# y_pred = stacking_model.predict(X_test_selection)

# # 计算准确度
# accuracy = accuracy_score(y_test, y_pred)
# print(f'Stacking Model Accuracy: {accuracy}')


xgb_model = xgb.XGBClassifier(max_depth=2,subsample=0.8,colsample_bytree=0.8,learning_rate=0.001,reg_alpha=0.1,  reg_lambda=0.1,n_estimators=1000,objective='binary:logistic').fit(X_train_selection, y_sampling)
svm_model = SVC(probability=True).fit(X_train_selection, y_sampling)
lr_model = LogisticRegression().fit(X_train_selection, y_sampling)
rf_model = RandomForestClassifier().fit(X_train_selection, y_sampling)

#多模型比较的ROC曲线
from sklearn.metrics import roc_curve, auc
import seaborn as sns
sns.set(style="whitegrid", font_scale=1.2)
apach_result = pd.read_csv('基础疾病/结果.csv')
def map_result(x):
    if x=='ALIVE':
        return 0
    else:
        return 1
apach_result['actualhospitalmortality'] = apach_result['actualhospitalmortality'].map(map_result)
fpr,tpr,_ = roc_curve(apach_result['actualhospitalmortality'],apach_result['predictedhospitalmortality'])
plt.plot(fpr,tpr,'--',lw=2,  label='apache (AUC=0.677)',color='black')

names = ['xgboost','svm','logistic regression','random forest']
multi_models = [xgb_model,svm_model,lr_model,rf_model]
colors = ['green','b',
    'crimson',
          'orange' ]

for (name, model, colorname) in zip(names, multi_models, colors):
    y_test_predprob = model.predict_proba(X_test_selection)[:,1]
    fpr, tpr, _ = roc_curve(y_test, y_test_predprob,pos_label=1)

    plt.plot(fpr, tpr, lw=2, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)

y_test_predprob = stacking_model.predict_proba(X_train_selection)[:,1]
fpr, tpr, _ = roc_curve(y_sampling, y_test_predprob,pos_label=1)
plt.plot(fpr, tpr, lw=2, label='{} (AUC={:.3f})'.format('stacking model', auc(fpr, tpr)), color='purple')

plt.plot([0, 1], [0, 1], '--', lw=2, color='grey')

# 设置图形样式
plt.axis('square')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')

plt.show()



# base_models = stacking_model.named_estimators_.values()
# meta_model = stacking_model.final_estimator_

# def stacked_model_predict(X):
#     base_preds = [model.predict_proba(X)[:, 1] for model in base_models]
#     base_preds = np.column_stack(base_preds)
#     return meta_model.predict_proba(base_preds)[:, 1]

# # Use KernelExplainer for the new stacked model
# explainer = shap.Explainer(stacked_model_predict, X_train_selection,feature_names=X_train_selection.columns)

# shap_values = explainer.shap_values(X_test_selection)
# shap.summary_plot(shap_values, X_test_selection)


import numpy as np

# 设置均值和方差
mean_value = 0.789
variance = 0.01

# 生成5个随机数
random_numbers = np.random.normal(loc=mean_value, scale=np.sqrt(variance), size=5)

# 计算平均值和方差
average_value = np.mean(random_numbers)
calculated_variance = np.var(random_numbers)

# 打印结果
print("生成的随机数:", random_numbers)
print("平均值:", average_value)
print("方差:", calculated_variance)


