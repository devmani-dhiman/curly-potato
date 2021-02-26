import pandas as pd
from scipy.sparse import data
import streamlit as st
from sklearn import datasets
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

def get_datasets(select_dataset):
    """
    Load the appropriate dataset according to the dataset selected by the user in the sidebar.

    """
    if select_dataset == 'Iris Dataset':
        dataset = datasets.load_iris()
    elif select_dataset == 'Breast Cancer':
        dataset = datasets.load_breast_cancer()
    else:
        dataset = datasets.load_wine()

    X = dataset.data
    y = dataset.target

    df = pd.DataFrame(X, columns=dataset.feature_names , index=None)
    df['Type'] = dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)
    return X_train, X_test, y_train, y_test,df,dataset.target_names


def clf_parameter(classifier_name):
    """
    To select the paramters according to the classifier selected.
    """
    param = dict()
    if classifier_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        param['K'] = K
    elif classifier_name == "SVM":
        C = st.sidebar.slider("C", 0.1, 10.0)
        param['C'] = C
    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 15)
        n_estimator = st.sidebar.slider("N estimator", 1, 100)
        param['max_depth'] = max_depth
        param['n_estimators'] = n_estimator
    return param


def get_classifier(classifier_name, params):
    """
    This programs trains the classifier on selected params in the above function.
    """
    if classifier_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif classifier_name == "SVM":
        clf = SVC(C=params['C'])
    else:
        clf = RandomForestClassifier(max_depth=params['max_depth'], n_estimators=params['n_estimators'],
                                     random_state=42)
    return clf

def getPCA(df):
    """
    To Draw PCA with 2 components
    """
    pca = PCA(n_components = 2)
    x_pca = pca.fit_transform(df.loc[:,df.columns != 'Type'])

    df['pca-1'] = x_pca[:, 0]
    df['pca-2'] = x_pca[:, 1]
    return df

st.title("Using Streamlit for creating Web apps")

st.write("""
### Using different algorithms to check on different datasets to check which performs better

The algorithms used are:
    
    1. KNN
    2. SVD
    3. Random Forest
    
The Datasets used are:

    1. Iris Dataset
    2. Breast Cancer Dataset 
    3. Wine Dataset   

""")

select_dataset = st.sidebar.selectbox(
    'Choose one of the following',
    ('Iris Dataset', 'Breast Cancer', 'Wine')
)

add_classifier = st.sidebar.selectbox(
    'Choose one of the following',
    ('KNN', 'SVM', 'Random Forest')
)


X_train, X_test, y_train, y_test, df, classes = get_datasets(select_dataset)
st.write("Selected data set is", format(select_dataset))
st.write("Shape of the dataset is", format(X_train.shape))
st.write("Number of unique features in the selected training dataset are ", format(len(np.unique(y_train))))

param = clf_parameter(add_classifier)

clf = get_classifier(add_classifier, param)

st.write("""
### Checking the accuracy of models on different datasets
""")

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.write("Classifier = ", add_classifier)
st.write(f"accuracy = {accuracy}")

st.write("""
#### Ploting the main features
""")

df = getPCA(df)
fig = plt.figure(figsize=(16,10))
sns.scatterplot(
    x='pca-1', y='pca-2',
    data= df,
    palette=sns.color_palette("hls", len(classes)),
    legend="full"
)
plt.xlabel('PCA One')
plt.ylabel('PCA Two')
plt.title("2-D PCA Visualization")
st.pyplot(fig)
