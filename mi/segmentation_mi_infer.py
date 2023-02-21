import pickle
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
import os
import pandas as pd


def get_result_table(y_true, y_pred, results=[]):
    conf_mat = confusion_matrix(y_true, y_pred)
    # https://towardsdatascience.com/understanding-the-confusion-matrix-from-scikit-learn-c51d88929c79
    TN, FP, FN, TP = conf_mat[0][0], conf_mat[0][1], conf_mat[1][0], conf_mat[1][1]
    TPR = (TP / (TP + FN)) # Sensitivity  , Recall
    TNR = (TN / (TN + FP)) # Specificity
    FPR = 1 - TNR # False alarm
    PPV = (TP / (TP + FP)) # Precision
    F1 = (2 * TP) / (2 * TP + FN + FP) # F1-score
    F2 =  (5 * TP) / (5 * TP + 4 * FN + FP) # F2-score
    ACC = (TP + TN) / (TP + TN + FN + FP)
    AUC = roc_auc_score(y_true, y_pred)
    return [TPR, TNR, FPR, PPV, F1, F2, ACC, AUC]
    # return [TPR, TNR, F1]
    # results.append([TPR, TNR, FPR, PPV, F1, F2, ACC, AUC])
    
def get_threshod(y_test, predicted_proba):
    false_pos_rate, true_pos_rate, proba = roc_curve(y_test, predicted_proba[:, -1])
    false_true_pts = np.dstack((false_pos_rate, true_pos_rate)).reshape(-1, 2)
    
    optimal_proba_cutoff_euclid = sorted(list(zip(np.sum(([0.0, 1.0] - false_true_pts)**2, axis=1), proba)), key=lambda i: i[0])[0][1]
    roc_predictions_euclid = [1 if i >= optimal_proba_cutoff_euclid else 0 for i in predicted_proba[:, -1]]
    
    optimal_proba_cutoff_ham = sorted(list(zip(np.abs(true_pos_rate - false_pos_rate), proba)), key=lambda i: i[0], reverse=True)[0][1]
    roc_predictions_ham = [1 if i >= optimal_proba_cutoff_ham else 0 for i in predicted_proba[:, -1]]
    
    return optimal_proba_cutoff_euclid, optimal_proba_cutoff_ham

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def get_model_and_params(pos_weights=0.1):
    clf1 = RandomForestClassifier(random_state=42)
    param1 = {}
    param1['classifier__n_estimators'] = [10, 50, 100, 250]
    param1['classifier__max_depth'] = [5, 10, 20]
    param1['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}, {0: 1, 1: pos_weights}]
    param1['classifier'] = [clf1]

    clf2 = SVC(probability=True, random_state=42)
    param2 = {}
    param2['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2]
    param2['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}, {0: 1, 1: pos_weights}]
    param2['classifier'] = [clf2]

    clf3 = LogisticRegression(random_state=42, max_iter=100)
    param3 = {}
    param3['classifier__C'] = [10**-2, 10**-1, 10**0, 10**1, 10**2] # [0.01, 0.1, 1, 10, 100] np.logspace(-4, 4, 20),
    param3['classifier__penalty'] = ['l1', 'l2'] # , 'l2'
    param3['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}, {0: 1, 1: pos_weights}]
    param3['classifier__solver'] = ['liblinear']
    param3['classifier'] = [clf3]

    clf4 = DecisionTreeClassifier(random_state=42)
    param4 = {}
    param4['classifier__max_depth'] = [5,10,25,None]
    param4['classifier__min_samples_split'] = [2,5,10]
    param4['classifier__class_weight'] = [None, {0:1,1:5}, {0:1,1:10}, {0:1,1:25}, {0: 1, 1: pos_weights}]
    param4['classifier'] = [clf4]

    clf5 = KNeighborsClassifier()
    param5 = {}
    param5['classifier__n_neighbors'] = [2,5,10,25,50]
    param5['classifier'] = [clf5]
    models = [clf1, clf2, clf3, clf4, clf5] # , clf6, clf7, clf8, clf9, clf10, clf11, clf12, clf13, clf14, clf15, clf16, clf17, clf18, clf19]

    pipelines = [ Pipeline([('classifier', f)]) for f in models  ]
    params = [ [param1], [param2], [param3], [param4], [param5]] #/ , [param6], [param7], [param8], [param9], [param10], [param11], [param12], [param13], [param14], [param15], [param16], [param17], [param18], [param19] ]
    return models, pipelines, params

columns_name = ['Seg_model', 'Classification_model', 'Model_Arch', 'Fold', 'Epoch', 'Train_IOU', 'Test_IOU', '#Train', '#Test', 'Sensitivity', 'Specificity', 'False_alarm', 'Precision', 'F1_score', 'F2_score', 'ACC', 'AUC']
results_df = {k:[] for k in columns_name}

def get_predict_acc(X_train, y_train, X_test, y_test, model_dl_path):
    
    global results_df
    
    ff = model_dl_path.split('/')[-1][:-8].split('__')
    mn = ff[0]
    ma = ff[1]
    fold = ff[3]
    epoch = ff[4]
    
    
    model_arch = f'{mn}__{ma}'
    
    train_acc = float(ff[5])
    test_acc = float(ff[6])
    
    
    
    models, pipelines, params = get_model_and_params()
    
        
    for idx, (pipeline, param) in enumerate(zip(pipelines, params)):
        # if idx == 2:
        #     break
        model = GridSearchCV(pipeline, param, cv=5, n_jobs=-1, scoring='roc_auc').fit(X_train, y_train)
        
        model_name = type(pipeline['classifier']).__name__
        
        # result = fit_model_and_thresh_hold(X_train, y_train, X_test, y_test, model_gs, is_cutoff)
        # print(f"Process to model idx: {idx} name: {model_name}")
    
        pred_train = model.predict_proba(X_train)
        pred_test = model.predict_proba(X_test)
         
        cutoff_euclid, cutoff_ham = get_threshod(y_train, pred_train)
        # cutoff_euclid, cutoff_ham = get_threshod(y_test, pred_test)
        
        # print("cutoff_euclid: {}".format(cutoff_euclid))
        
        roc_pred_eu = [1 if i >= cutoff_euclid else 0 for i in pred_test[:, -1]]
        
        result =  get_result_table(y_test, roc_pred_eu)        
        # print(result)
        for idc, cn in enumerate(columns_name):
            if idc == 0:
                results_df[cn].append(os.path.basename(model_dl_path))
            elif idc == 1:
                results_df[cn].append(model_name)
            elif idc == 2:
                results_df[cn].append(model_arch)
            elif idc == 3:
                results_df[cn].append(fold)
            elif idc == 4:
                results_df[cn].append(epoch)
            elif idc == 5:
                results_df[cn].append(train_acc)
            elif idc == 6:
                results_df[cn].append(test_acc)
            elif idc == 7:
                results_df[cn].append(len(y_train))
            elif idc == 8:
                results_df[cn].append(len(y_test))
                
            else:
                results_df[cn].append(result[idc - 9])
    
    # print(pd.DataFrame(results_df, index=None))
    
    

def get_data_motion(folder_npz='/home/vishc2/tuannm/echo/vishc-echo/models/embed_json'): # fn='/home/vishc2/tuannm/echo/vishc-echo/models/embed_json/Linknet__resnet18__050__000__043__0.9579__0.8737.pth.npz'):
    import glob
    files_npz = glob.glob(os.path.join(folder_npz, '**', '*.npz'), recursive=True)
    
    for idf, fn in enumerate(files_npz):
        print("process to file: {} / {}".format(idf + 1, len(files_npz)))
        # if idf == 5:
        #     break
        data = pickle.load(open(fn, 'rb'))
        data_train = data['train']
        data_test = data['test']
        
        X_train, y_train, X_test, y_test = [], [], [], []
        for fp, d_value in data['train'].items():
            X_train.append(d_value[0])
            y_train.append(d_value[1])
        
        for fp, d_value in data['test'].items():
            X_test.append(d_value[0])
            y_test.append(d_value[1])
        
        X_train, y_train, X_test, y_test = np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)
        # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        get_predict_acc(X_train, y_train, X_test, y_test, fn)
        break
    global results_df
    
    # df = pd.DataFrame(results_df, index=None)
    # df.to_excel('./xlsx/mi_embed_json.xlsx')
    
    # return data


if __name__ == '__main__':
    data = get_data_motion()