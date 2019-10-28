# coding:utf-8
import numpy as np
import pandas as pd
import pickle
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')
# read data
train = pd.read_csv("./data/titanic/train.csv")
test = pd.read_csv("./data/titanic/test.csv")


def clean_data(titanic):  # 填充空数据 和 把string数据转成integer表示
    titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
    # child
    titanic["child"] = titanic["Age"].apply(lambda x: 1 if x < 15 else 0)

    # sex
    titanic["sex"] = titanic["Sex"].apply(lambda x: 1 if x == "male" else 0)

    titanic["Embarked"] = titanic["Embarked"].fillna("S")

    # embark
    def getEmbark(Embarked):
        if Embarked == "S":
            return 1
        elif Embarked == "C":
            return 2
        else:
            return 3

    titanic["embark"] = titanic["Embarked"].apply(getEmbark)

    # familysize
    titanic["fimalysize"] = titanic["SibSp"] + titanic["Parch"] + 1

    # cabin
    def getCabin(cabin):
        if cabin == "N":
            return 0
        else:
            return 1

    titanic["cabin"] = titanic["Cabin"].apply(getCabin)

    # name
    def getName(name):
        if "Mr" in str(name):
            return 1
        elif "Mrs" in str(name):
            return 2
        else:
            return 0

    titanic["name"] = titanic["Name"].apply(getName)

    titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

    return titanic


# 对数据进行清洗
features = ["Pclass", "sex", "child", "fimalysize", "Fare", "embark", "cabin"]
train_data = clean_data(train)
X = train_data[features]
y= train_data["Survived"]
xtrain,xtest,ytrian,ytest=train_test_split(X,y, test_size=0.33, random_state=42)


clf = XGBClassifier(learning_rate=0.1,n_jobs=-1, silent=True, objective='binary:logistic',subsample=0.7,colsample_bytree=0.7)
param_test = {
    'n_estimators': range(40, 50, 2),
    'max_depth': range(2, 5, 1),
    'reg_lambda':[0.1,1,5,10],
    'min_child_weight':[0.5,1,3,5]
}
grid_search = GridSearchCV(estimator=clf, param_grid=param_test, scoring='accuracy', cv=5)
grid_search.fit(xtrain, ytrian)
model = grid_search.best_estimator_
print(grid_search.best_params_)
print(grid_search.best_score_)

with open('./save/titanic/xgboost_titanic.pickle','wb') as f:
    pickle.dump(model,f)
# with open('./save/titanic/xgboost_titanic.pickle','rb') as f:
#     model=pickle.load(f)
ytest_predict=model.predict(xtest)
print(model.score(xtest,ytest)) ##0.8169491525423729
print(classification_report(ytest,ytest_predict))

##训练集auc
ytrain_predict=model.predict(xtrain)
print(roc_auc_score(ytrian,ytrain_predict))

##测试集auc
print(roc_auc_score(ytest,ytest_predict))

##全部数据集
y_predict=model.predict(X)
print(roc_auc_score(y,y_predict))