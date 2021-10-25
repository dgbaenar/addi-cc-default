import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
import shap
import joblib

# Options
# version = 'v3'
train_dataset_path = './data/train.csv'
test_dataset_path = './data/test.csv'
explainer_path = './data/models/shap_explainer.joblib.dat'
model_path = './data/models/model.joblib.dat'
scores_file = './data/metrics/scores.json'
threshold_file = './data/metrics/threshold.json'
metric_name = "DEFAULT PAYMENT NEXT MONTH"

# A parameter grid for XGBoost
params = {
    'min_child_weight': [1, 3, 5],
    'gamma': [0.5],
    'subsample': [0.5],
    'colsample_bytree': [0.5],
    'colsample_bylevel': [0.5],
    'max_depth': [5, 10, 15, 20],
    'learning_rate': [0.1, 0.01],
    'max_delta_step': [0.5],
    'n_estimators': [10, 50, 100, 200],
    'reg_alpha': [0.5],
    'reg_lambda': [0.5]
}


def eval_metrics(actual, pred):
    auc = metrics.roc_auc_score(actual, pred)
    return auc


def main():

    # Initialize XGB models
    clf = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        use_label_encoder=False,
        seed=42
    )

    # GridSearch
    gsearch = GridSearchCV(
        estimator=clf,
        param_grid=params,
        scoring='roc_auc',
        cv=5
    )

    # Read Train/Test
    train = pd.read_csv(train_dataset_path)
    test = pd.read_csv(test_dataset_path)

    # Train
    y_train = train.pop(metric_name)
    X_train = train

    y_test = test.pop(metric_name)
    X_test = test

    # Train
    print('Training...')

    models = gsearch.fit(X_train, y_train)
    print(models.best_score_)
    model = models.best_estimator_

    # Check the accuracy
    print("Train score: " + str(model.score(X_train, y_train)))
    print("Test score: " + str(model.score(X_test[X_train.columns], y_test)))

    # Save models
    print('Exporting models...')
    joblib.dump(model, model_path)

    # Confusion Matrix Train
    y_pred = model.predict_proba(X_train)[:, 1]
    pd.cut(y_pred, 10).value_counts().plot(kind='barh')
    plt.savefig('./data/img/proba_dist_train.png')
    plt.show()
    precision, recall, thresholds = metrics.precision_recall_curve(
        y_train, y_pred)
    f1_scores = 2 * recall * precision / (recall + precision + 0.001)
    threshold = float(thresholds[np.argmax(f1_scores)])
    print("Threshold: %s" % threshold)
    roc_auc = eval_metrics(y_train, y_pred)
    print("AUC: %s" % roc_auc)
    y_pred = (y_pred >= threshold).astype(int)
    print("Confusion matrix train set:")
    print(metrics.confusion_matrix(y_train, y_pred))

    # Confusion Matrix Test
    y_pred = model.predict_proba(X_test[X_train.columns])[:, 1]
    pd.cut(y_pred, 10).value_counts().plot(kind='barh')
    plt.savefig('./data/img/proba_dist_test.png')
    plt.show()
    precision, recall, thresholds = metrics.precision_recall_curve(
        y_test, y_pred)
    f1_scores = 2 * recall * precision / (recall + precision)
    f1_score = f1_scores[np.argmax(f1_scores)]
    threshold = float(thresholds[np.argmax(f1_scores)])
    precision = float(precision[np.argmax(f1_scores)])
    recall = float(recall[np.argmax(f1_scores)]) 
    print("Threshold: %s" % threshold)
    roc_auc = eval_metrics(y_test, y_pred)
    print("AUC: %s" % roc_auc)
    # threshold = optimize_th(y_test, y_pred)
    y_pred = (y_pred >= threshold).astype(int)
    cm_test = metrics.confusion_matrix(y_test, y_pred)
    print("Confusion matrix test set:")
    print(cm_test)

    # Save classes
    classes = pd.DataFrame(y_pred).join(y_test)
    classes.columns = ['predicted', 'actual']
    classes.to_csv('./data/classes.csv', index=False)

    with open(scores_file, 'w') as fd:
        json.dump({'roc_auc': roc_auc,
                   'precision': precision,
                   'recall': recall,
                   'f1_score': f1_score}, fd, indent=4)

    with open(threshold_file, 'w') as fd:
        json.dump({'threshold': threshold}, fd, indent=4)

    # Explainability
    print('Explainability...')
    explainer = shap.TreeExplainer(model)
    joblib.dump(explainer, explainer_path)
    shap_values = explainer(X_test[X_train.columns])

    # Save confusion matrix
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm_test)
    disp.plot()
    plt.savefig('./data/img/confusion_matrix.png')
    plt.show()

    # Summarize the effects of all the features
    shap.plots.beeswarm(shap_values, plot_size=(15, 8), max_display=20, show=False); plt.subplots_adjust(left=0.3)
    plt.savefig('./data/img/shap.png')
    plt.close()


if __name__ == "__main__":
    main()
