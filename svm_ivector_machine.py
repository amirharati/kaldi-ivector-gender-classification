# svm_ivector_machine.py
# Amir Harati May 2019
"""
    Test extracted ivectors using SVM machine.
"""

from read_kaldi_output import read_kaldi_output
from sklearn import svm
import sklearn
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import precision_recall_fscore_support
import scipy.stats
import sklearn.metrics

def main():

    train_keys, train_data = read_kaldi_output("kaldi_outputs/plda_train_ivec.txt")
    eval_keys, eval_data = read_kaldi_output("kaldi_outputs/plda_eval_ivec.txt")

    train_labs = []
    for tk in train_keys:
        if tk.split("_")[0] == "male":
            l = 0
        else:
            l = 1
        train_labs.append(l)
    
    eval_labs = []
    for ek in eval_keys:
        if ek.split("_")[0] == "male":
            l = 0
        else:
            l = 1
        eval_labs.append(l)


    model = svm.SVC(kernel='rbf', probability=True)
    #print(train_data)
    model.fit(train_data, train_labs)

    z = model.predict_proba(eval_data)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(eval_labs, z[:,1], pos_label=1)
    auc = sklearn.metrics.auc(fpr, tpr)
    
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    fpr_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    tpr_eer = 1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    specificity, sensitivity = 1-fpr_eer, tpr_eer
    y_pred_bin = [float(y >= eer_threshold)  for y in list(z[:, 1])]
    cm = confusion_matrix(eval_labs, y_pred_bin)
    total = 1.0 * sum(sum(cm))
    accuracy = (cm[0,0] + cm[1,1]) / total
    precision, recall, f1, support = precision_recall_fscore_support(eval_labs, y_pred_bin)



    print("AUC: ", auc)
    print("EER specificity, sensitivity:", specificity, sensitivity)
    print("EER cm:", cm)
    print("EER accuracy:", accuracy)
    print("EER precision, recall, f1, support:", precision, recall, f1, support)

    index = 0
    print("error cases")
    for l, yb, s in zip(eval_labs, y_pred_bin, list(z[:, 1])):
        if l != yb:
            print(eval_keys[index]," score (post. being class female): ", s)
        index += 1

if __name__ == "__main__":
    main()
