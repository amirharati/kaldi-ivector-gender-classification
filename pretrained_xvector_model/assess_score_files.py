# assess_score_files.py
# Amir Harati May 2019
"""
    read score files from Kaldi and compute metrics.
"""
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


model0 = "female"

files2score = {}
files2label = {}
def main():
    lines = [line.strip() for line in open("xvectors/scores/plda_scores")]
    for line in lines:
        model, fi, score = line.split()
        lab = fi.split("_")[0]
        if lab == model0:
            lab_n = 1
        else:
            lab_n = 0
        
        files2label[fi] = lab_n

        if fi not in files2score:
            if model == model0:
                files2score[fi] = float(score)
            else:
                files2score[fi] = -float(score)
        else:
            if model == model0:
                files2score[fi] += float(score)
            else:
                files2score[fi] += -float(score)

    labels = []
    scores = []
    for key in files2label:
        l = files2label[key]
        s = files2score[key]
        labels.append(l)
        scores.append(s)

    fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, scores, pos_label=1)
    auc = sklearn.metrics.auc(fpr, tpr)
    
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    fpr_eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    tpr_eer = 1 - fnr[np.nanargmin(np.absolute((fnr - fpr)))]

    specificity, sensitivity = 1-fpr_eer, tpr_eer
    y_pred_bin = [float(y >= eer_threshold)  for y in list(scores)]
    cm = confusion_matrix(labels, y_pred_bin)
    total = 1.0 * sum(sum(cm))
    accuracy = (cm[0,0] + cm[1,1]) / total
    precision, recall, f1, support = precision_recall_fscore_support(labels, y_pred_bin)



    print("AUC: ", auc)
    print("EER specificity, sensitivity:", specificity, sensitivity)
    print("EER cm:", cm)
    print("EER accuracy:", accuracy)
    print("EER precision, recall, f1, support:", precision, recall, f1, support)
    

if __name__ == "__main__":
    main()
