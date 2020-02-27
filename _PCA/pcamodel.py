from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd
df = pd.read_csv('iris.data',header=0)
df.columns=['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid', 'label']
# print(df)
def noPCA():
    """
    降维前
    """
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                            ('blue', 'red', 'green')):
         plt.scatter(df[df['label']==lab]['sepal_len'],
                    df[df['label']==lab]['sepal_wid'],
                    label=lab,
                    c=col)
    plt.xlabel('sepal_len')
    plt.ylabel('sepal_wid')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

lower_dim = PCA(n_components=4)
lower_dim.fit(df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']].values)
Y=lower_dim.transform(df[['sepal_len', 'sepal_wid', 'petal_len', 'petal_wid']].values)

# if __name__ == '__main__':
    # noPCA()

df1=pd.DataFrame({'Component_1':Y[:,0],"Component_2":Y[:,1],"label":df['label']})
def pca():
    plt.figure(figsize=(6, 4))
    for lab, col in zip(('Iris-setosa', 'Iris-versicolor', 'Iris-virginica'),
                        ('blue', 'red', 'green')):
        plt.scatter(df1[df1['label']==lab]['Component_1'],
                    df1[df1['label']==lab]['Component_2'],
                    label=lab,
                    c=col)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='lower center')
    plt.tight_layout()
    plt.show()
if __name__ == '__main__':
    pca()
    # noPCA()
