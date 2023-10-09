import pandas as pd 
import numpy as np 
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve,f1_score, auc
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font',family='Times New Roman')
sns.set(style="whitegrid", font_scale=1.2)

#为了同时最大化敏感度和特异度，我们需要先找到最优的阈值
def find_optimal_cutoff(tpr,fpr,threshold):
    optimal_idx = np.argmax(tpr - fpr)
    optimal_treshold = threshold[optimal_idx]
    return optimal_treshold

def best_confusion_matrix(y_test,y_test_predprob):
    fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)
    cutoff = find_optimal_cutoff(tpr, fpr, thresholds)
    y_pred = list(map(lambda x: 1 if x >= cutoff else 0, y_test_predprob))
    TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()
    return cutoff, TN, FN, FP, TP

def evaluation(clf, X_train,y_train,X_test,y_test,modelname,digits):
    y_train_predprob = clf.predict_proba(X_train)[:, 1]
    train_auc = round(roc_auc_score(y_train, y_train_predprob), digits)

    y_test_predprob = clf.predict_proba(X_test)[:,1]
    test_auc = round(roc_auc_score(y_test,y_test_predprob), digits)

    #train_cutoff, TN1, FN1, FP1, TP1 = best_confusion_matrix(y_train,y_train_predprob)
    test_cutoff, TN2, FN2, FP2, TP2 = best_confusion_matrix(y_test,y_test_predprob)

     #Sen Spe
    best_recall,best_prec = round(TP2/(TP2+FN2),digits), round(TN2/(FP2+TN2),digits)
 
    #PPV NPV
    npv,ppv = round(TN2/(FN2+TN2),digits),round(TP2/(TP2+FP2),digits)
    
    #PLR NLR
    plr,nlr = round((TP2/(TP2+FN2))/(FP2/(FP2+TN2)),digits),round((FN2/(TP2+FN2))/(TN2/(FP2+TN2)),digits)

    y_test_pred=list(map(lambda x:1 if x>=test_cutoff else 0,y_test_predprob))
    f1 = round(f1_score(y_test, y_test_pred),digits)
    
    #Youden Index
    youden = round(TP2/(TP2+FN2)+TN2/(FP2+TN2)-1,digits)

    #MCC
    mcc = round(metrics.matthews_corrcoef(y_test, y_test_pred),digits)
    
    #Kappa
    kappa =round(metrics.cohen_kappa_score(y_test_pred, y_test),digits)
  
    eval_df = {'Model': ['train_auc','test_auc','specificity','sensitivity','F1','Youden Index','MCC','Kappa','npv','ppv','plr','nlr'],
     modelname: [train_auc, test_auc, best_prec,best_recall,f1,youden,mcc,kappa,npv,ppv,plr,nlr]}
    eval_df = pd.DataFrame(data=eval_df)

    return eval_df


def multi_models_roc(names, multi_models, colors, X_test, y_test, save=True, dpin=100):
    """
    将多个机器模型的roc图输出到一张图上
    """
    plt.figure(figsize=(20,20),dpi = dpin)  
    apach_result = pd.read_csv('基础疾病/结果.csv')
    def map_result(x):
        if x=='ALIVE':
            return 0
        else:
            return 1
    apach_result['actualhospitalmortality'] = apach_result['actualhospitalmortality'].map(map_result)
    fpr,tpr,_ = roc_curve(apach_result['actualhospitalmortality'],apach_result['predictedhospitalmortality'])
    plt.plot(fpr,tpr,'--',lw=5,  label='{} (AUC={:.3f}'.format('apache',auc(fpr,tpr)),color='black')
    plt.legend(loc='lower right',fontsize=20)   

    for (name, model, colorname) in zip(names, multi_models, colors):
        y_test_predprob = model.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_test_predprob,pos_label=1)

        plt.plot(fpr, tpr, lw=2, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
        plt.plot([0,1],[0,1],'--', lw=2, color= 'grey')
        plt.axis('square')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.xlabel('False Positive Rate',fontsize=14)
        plt.ylabel('True Positive Rate',fontsize=14)
        plt.title('Receiver Operating Characteristic (ROC) Curve',fontsize=16)
        plt.legend(loc='lower right',fontsize=12)   
    
    if save:
        plt.savefig('multi_models_roc.png')
    return plt





