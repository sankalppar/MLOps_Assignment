import dvc.api
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import json

#with dvc.api.open(repo = "https://github.com/sankalppar/MLOps_Assignment", path = "data/creditcard.csv") as fd:
#    df = pd.read_csv(fd)
#    train = df.sample(frac=0.8,random_state=200)
#    test = df.drop(train.index)
#    train.to_csv("../data/processed/train.csv")
#    test.to_csv("../data/processed/test.csv")
df_train = pd.read_csv("../data/processed/train.csv")
clf = DecisionTreeClassifier(criterion = "entropy")
clf = clf.fit(df_train.drop("Class", axis = 1), df_train["Class"])
joblib.dump(clf, "../models/model.pkl")
df_test = pd.read_csv("../data/processed/test.csv")
pred = clf.predict(df_test.drop("Class", axis = 1))
acc = metrics.accuracy_score(df_test["Class"], pred)
f1_score = metrics.f1_score(df_test["Class"], pred, average = "weighted")
metric = {'accuracy' : acc, 'F1 Score' : f1_score}
metric_file = open("../metrics/acc_f1.json", "w")
json.dump(metric, metric_file)
metric_file.close()

