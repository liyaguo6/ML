import numpy as np
import pandas as pd
from scipy import linalg

def cal_mean(X,y):
    """
    计算不同类别的均值
    """
    mean_vectors = {}
    classs = np.unique(y)
    for i in list(classs):
        mean_vectors[i] =np.mean(X[y==i],axis=0)
    return mean_vectors

def cal_Sw(X,y,mean_vectors):
    """
    计算类内散步举证
    """
    n_classes = X.shape[1]
    # Sw = np.zeros((n_classes, n_classes))
    # for cl, mv in mean_vectors.items():
    #     class_sc_mat = np.zeros((n_classes, n_classes))  # scatter matrix for every class
    #     for row in X[y == cl]:
    #         row, mv = row.reshape(row.shape[0], 1), mv.reshape(n_classes, 1)  # make column vectors
    #         class_sc_mat += (row - mv).dot((row - mv).T)
    #     Sw+= class_sc_mat
    Sw = np.zeros((n_classes, n_classes))
    for cl, mv in mean_vectors.items():
        class_sc_mat = (X[y == cl]-mean_vectors[cl]).T
        Sw+= class_sc_mat.dot(class_sc_mat.T)
    return Sw

def cal_Sb(X,y,cls_mean_vectors,overall_mean):
    """
    计算內间散布矩阵
    """
    Sb = np.zeros((X.shape[1],X.shape[1]))
    for i, mean_vec in cls_mean_vectors.items():
        n = X[y == i , :].shape[0]
        mean_vec = mean_vec.reshape(mean_vec.shape[0], 1)  # make column vector
        overall_mean = overall_mean.reshape(overall_mean.shape[0], 1)  # make column vector
        Sb += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)
    return Sb

def cal_eigVec_eigVal(Sw,Sb):
    """
    计算Sw^(-1)*Sb特征值和特征向量
    """
    # eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(Sw).dot(Sb))
    eig_vals, eig_vecs = linalg.eig(np.mat((np.linalg.inv(Sw).dot(Sb))))
    return eig_vals, eig_vecs


def cal_y(X,eig_pairs,n_compoments=1):
    """
    降维计算
    """
    W = []
    for val,vec in eig_pairs[:n_compoments]:
        W.append(vec)
    WT = np.array(W)
    y =WT.dot(X.T)
    return y.T

from matplotlib import pyplot as plt

def plot_step_lda(X_lda):
    """
    画图
    """

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X_lda[:,0].real[y == label],
                y=X_lda[:,1].real[y == label],
                marker=marker,
                color=color,
                alpha=0.5,
                label=label_dict[label]
                )

    plt.xlabel('LD1')
    plt.ylabel('LD2')

    leg = plt.legend(loc='upper right', fancybox=True)
    leg.get_frame().set_alpha(0.5)
    plt.title('LDA: Iris projection onto the first 2 linear discriminants')

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.show()





if __name__ == '__main__':

    feature_dict = {i: label for i, label in zip(
        range(4),
        ('sepal length in cm',
         'sepal width in cm',
         'petal length in cm',
         'petal width in cm',))}
    df = pd.io.parsers.read_csv(
        filepath_or_buffer='https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data',
        header=None,
        sep=',',
    )
    df.columns = [l for i, l in sorted(feature_dict.items())] + ['class label']
    df.dropna(how="all", inplace=True)  # to drop the empty line at file-end
    from sklearn.preprocessing import LabelEncoder

    X = df[['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm']].values
    y = df['class label'].values

    enc = LabelEncoder()
    label_encoder = enc.fit(y)
    y = label_encoder.transform(y) + 1

    cls_mean_vectors=cal_mean(X,y)
    # # print(y.shape)
    print("cls_mean_vectors:\n",cls_mean_vectors)
    Sw=cal_Sw(X, y, cls_mean_vectors)
    print("SW:\n",Sw)
    overall_mean = np.mean(X,axis=0)
    Sb=cal_Sb(X,y,cls_mean_vectors,overall_mean)
    print("Sb:\n",Sb)
    eig_vals, eig_vecs =cal_eigVec_eigVal(Sw=Sw,Sb=Sb)
    print("eig_vals:\n",eig_vals)
    print("eig_vecs:\n",eig_vecs)
    eig_pairs =[(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]
    eig_pairs = sorted(eig_pairs, key=lambda k: k[0], reverse=True)
    yt=cal_y(X,eig_pairs,2 )
    print("降维后yt:\n",yt[:3,:])
    print("#"*30)

    label_dict = {1: 'setosa', 2: 'versicolor', 3: "virginica"}
    plot_step_lda(yt)