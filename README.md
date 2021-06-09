# Thyroid Detection Web App

![thyroid_logo](https://user-images.githubusercontent.com/44118554/121289362-6fef3f80-c902-11eb-837f-37611aeb512f.jpg)

This app predicts the chances of having **Thyroid Disease** using user provided Medical Data. Training Data is obtained from Kaggle website , Training Dataset Link :
https://www.kaggle.com/kumar012/hypothyroid?select=hypothyroid.csv

## Model Pipeline

![download](https://user-images.githubusercontent.com/44118554/121290011-85b13480-c903-11eb-837b-a6d338da72a6.jpg)

### Model Building Steps :

#### Step 1 : Importing Libraries and Dataset : 

Important python libraries used were Scikit-Learn , Pandas , Numpy , Imblearn and  Streamlit . The Training dataset was obtaib=ned from Kaggle website 
Dataset Link : https://www.kaggle.com/kumar012/hypothyroid?select=hypothyroid.csv

#### Step 2 : Data Preprocessing : 

In this step the missing values were handled using KNN imputer and unwanted columns from the dataset were removed.

#### Step 3 : Exploratory Data Analysis :

In this step Feature Scaling , Normalization and Over Sampling(Imbalanced Data) of the data was done.

#### Step 4 : Data Clustering  :

**KNN Clustering algorithm** was used to divide the Data into multiple clusters and the optimal number of clusters was decided using elbow method.

![download (1)](https://user-images.githubusercontent.com/44118554/121291657-43d5bd80-c906-11eb-9caa-dfa7d9bc0c75.png)

![download](https://user-images.githubusercontent.com/44118554/121291553-16890f80-c906-11eb-9e72-7690169d31d9.png)

#### Step 5 : Model Training :

Each Cluster was trained on **Random Forest Classifier , Naive Bayes Classifier and Support Vector Classifier** and the optimal was choosen for each cluster.

#### Step 6 : Model Prediction and Evaluation and Hyper parameter tuning:

The model predictions were made  on the test dataset and  Hyper parameter tunong was done using Grid search Cv**ROC-AUC** score was used for evaluation of the model.

![Capture](https://user-images.githubusercontent.com/44118554/121291898-a333cd80-c906-11eb-9c04-e1b8def39f8e.PNG)

### Website Link : https://thyroid-prediction-streamlit.herokuapp.com/


