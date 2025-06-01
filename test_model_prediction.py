# Load and display Iris dataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

feature_names = iris.feature_names
target_names = iris.target_names

print("Feature names:", feature_names)
print("Target names:", target_names)
print("\nType of X is:", type(X))
print("\nFirst 5 rows of X:\n", X[:5])

# Split Iris dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=1)
print("X_train Shape:", X_train.shape)
print("X_test Shape:", X_test.shape)
print("Y_train Shape:", y_train.shape)
print("Y_test Shape:", y_test.shape)

# Label Encoding Example
categorical_feature = ['cat', 'dog', 'dog', 'cat', 'bird']
encoder = LabelEncoder()
encoded_feature = encoder.fit_transform(categorical_feature)
print("Encoded feature:", encoded_feature)

# One Hot Encoding Example
categorical_feature = np.array(['cat', 'dog', 'dog', 'cat', 'bird']).reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
encoded_feature = encoder.fit_transform(categorical_feature)
print("OneHotEncoded feature:\n", encoded_feature)

# Train Logistic Regression on Iris
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print("Logistic Regression model accuracy:", metrics.accuracy_score(y_test, y_pred))

# Sample Predictions
sample = [[3, 5, 4, 2], [2, 3, 5, 4]]
preds = log_reg.predict(sample)
pred_species = [iris.target_names[p] for p in preds]
print("Predictions:", pred_species)

# Import libraries for Framingham dataset
import pandas as pd
import pylab as pl
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

# Load and preprocess Framingham dataset
disease_df = pd.read_csv(r"D:\Project\NEW PROJECT\heart_disease_prediction\data\framingham.csv")
disease_df.drop(['education'], inplace=True, axis=1)
disease_df.rename(columns={'male': 'Sex_male'}, inplace=True)
disease_df.dropna(axis=0, inplace=True)

print(disease_df.TenYearCHD.value_counts())

X = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay', 'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

X = preprocessing.StandardScaler().fit(X).transform(X)

# Split Framingham dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=4)
print('Train set:', X_train.shape, y_train.shape)
print('Test set:', X_test.shape, y_test.shape)

# Plot class distribution
plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=disease_df, palette="BuGn_r")
plt.title("Heart Disease Distribution")
plt.show()

# Line plot of CHD occurrences
disease_df['TenYearCHD'].plot()
plt.title("Ten Year CHD Trend")
plt.xlabel("Index")
plt.ylabel("CHD (0 = No, 1 = Yes)")
plt.grid(True)
plt.show()

# Train Logistic Regression model
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)

# Accuracy and Confusion Matrix
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
print('Accuracy of the model is =', accuracy_score(y_test, y_pred))

print('The details for confusion matrix is =')
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data=cm, 
                           columns=['Predicted:0', 'Predicted:1'], 
                           index=['Actual:0', 'Actual:1'])

plt.figure(figsize=(8, 5))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Greens")
plt.title("Confusion Matrix")
plt.show()