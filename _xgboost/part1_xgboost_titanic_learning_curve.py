# coding:utf-8
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.metrics import classification_report
import warnings

warnings.filterwarnings('ignore')
# read data

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (5.0, 4.0)


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

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.7, 1., 15), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()  # 纵坐标反向
        plt.grid()  # 设置网格线

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"测试/交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff
# 对数据进行清洗
if __name__ == '__main__':

    train = pd.read_csv("./data/titanic/train.csv")
    test = pd.read_csv("./data/titanic/test.csv")
    features = ["Pclass", "sex", "child", "fimalysize", "Fare", "embark", "cabin"]
    train_data = clean_data(train)
    X = train_data[features]
    y= train_data["Survived"]

    with open('./save/titanic/xgboost_titanic.pickle','rb') as f:
        model=pickle.load(f)

    plot_learning_curve(model, u"学习曲线", X, y, cv=4)



