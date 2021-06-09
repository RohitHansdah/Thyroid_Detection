import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import time
import logging
from PIL import Image
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
logging.basicConfig(filename="log.txt",
                    format='%(asctime)s %(message)s',
                    filemode='w')
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.info("Logging test")
st.title("Thyroid Detection Web App")
image = Image.open('thyroid_logo.jpg')
st.image(image, use_column_width=True)
st.write("""
This app predicts the chances of having **Thyroid Disease** using user provided Medical Data.
Training Data is obtained from Kaggle website , Training Dataset Link :  
*** https://www.kaggle.com/kumar012/hypothyroid?select=hypothyroid.csv ***
""")
logger.info("importing test dataset")
uploaded_file = st.file_uploader("Choose a CSV file")
st.write(""" Please ** keep the column names in the CSV File ** as follows : 'age', 'sex', 'on thyroxine', 'query on thyroxine',
           'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
           'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
           'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U',
           'FTI', 'referral source' """)
st.header(""" Sample dataframe """)
test_df=pd.read_csv("test.csv")
st.write(test_df)
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.header("Uploaded dataframe")
    st.write(df)
    df.head()
    logger.info("data preprocessing in test dataset")
    df.info()
    df.describe()
    logger.info("Converting categorical data columns to binary in test dataset")
    def convert(col):
        for i in range(0,len(df[col])):
            if df[col][i]=='?':
                df[col][i]=np.nan
            elif df[col][i] =='F' or df[col][i]=='f' or df[col][i]=='N':
                df[col][i]=int(0)
            elif df[col][i] =='M' or df[col][i]=='t' or df[col][i]=='P':
                df[col][i]=int(1)
        if(col!="referral source"):
            df[col]=df[col].astype('float64')
    for x in df.columns:
        convert(x)
    df.head()
    logger.info("Handling missing data using KNN imputer in test dataset")
    imputer = KNNImputer(n_neighbors=5)
    df['age']=imputer.fit_transform(np.array(df['age']).reshape(-1, 1))
    df['sex']=imputer.fit_transform(np.array(df['sex']).reshape(-1, 1))
    df['TSH']=imputer.fit_transform(np.array(df['TSH']).reshape(-1, 1))
    df['T3']=imputer.fit_transform(np.array(df['T3']).reshape(-1, 1))
    df['T4U']=imputer.fit_transform(np.array(df['T4U']).reshape(-1, 1))
    df['FTI']=imputer.fit_transform(np.array(df['FTI']).reshape(-1, 1))
    df['TT4']=imputer.fit_transform(np.array(df['TT4']).reshape(-1, 1))
    logger.info("Encoding Categorical Data in test dataset")
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(df['referral source'])
    print(integer_encoded)
    onehot_encoder = OneHotEncoder(drop="first",sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    print(onehot_encoded)
    df['referral source1']=onehot_encoded[:,0]
    df['referral source2']=onehot_encoded[:,1]
    df['referral source3']=onehot_encoded[:,2]
    df['referral source4']=onehot_encoded[:,3]
    del df['referral source']
    df.head()
    logger.info("Exploratory Data Analysis in test dataset")
    sns.heatmap(df.corr())
    logger.info("Checking for skewed data in test dataset")
    sns.displot(df['T4U'])
    sns.displot(df['TSH'])
    sns.displot(df['T3'])
    sns.displot(df['FTI'])
    sns.displot(df['TT4'])
    sns.displot(df['age'])
    logger.info("Feature Scaling in test dataset")
    X = df.values
    print(X)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    logger.info("Normalizing Data in test dataset")
    X = preprocessing.normalize(X)
    logger.info("Data Clustering in test dataset")
    logger.info("Finding Number of clusters using elbow method in test dataset")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('The Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()
    logger.info("Training the K-Means model on the dataset in test dataset")
    num_of_clusters=3
    kmeans = KMeans(n_clusters = num_of_clusters, init = 'k-means++', random_state = 42)
    y_kmeans = kmeans.fit(X)
    logger.info("Adding clusters to the dataframe in test dataset")
    df1 = pd.DataFrame(X)
    df1.columns=['age', 'sex', 'on thyroxine', 'query on thyroxine',
           'on antithyroid medication', 'sick', 'pregnant', 'thyroid surgery',
           'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
           'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U',
           'FTI', 'referral source1', 'referral source2', 'referral source3',
           'referral source4']
    cluster=pd.DataFrame()
    cluster['Clusters']=kmeans.labels_
    df1.head()
    logger.info("Model Selection : Random forest VS Naive Bayes VS SVC in test dataset")
    logger.info("Reads the saved pretrained model")
    cluster0_model = pickle.load(open('cluster0_model.pkl', 'rb'))
    cluster1_model = pickle.load(open('cluster1_model.pkl', 'rb'))
    cluster2_model = pickle.load(open('cluster2_model.pkl', 'rb'))
    logger.info("Saving Predictions in a Dataframe")
    res=[]
    for i in range(0,len(df1)):
        if cluster['Clusters'][i] == 0 :
            arr = np.array(df1.iloc[i].values)
            arr = arr.reshape(1,-1)
            res.append(cluster0_model.predict(arr))
        elif cluster['Clusters'][i] == 1 :
            arr = np.array(df1.iloc[i].values)
            arr = arr.reshape(1,-1)
            res.append(cluster1_model.predict(arr))
        elif cluster['Clusters'][i] == 2 :
            arr = np.array(df1.iloc[i].values)
            arr = arr.reshape(1,-1)
            res.append(cluster2_model.predict(arr))
    st.header('Predictions')
    prediction=pd.DataFrame(res)
    st.write(prediction)
    logger.info("Successfully Finished the execution of App !!!")
