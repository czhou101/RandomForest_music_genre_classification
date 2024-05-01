import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import utils


train_split = 5
test_split = 5

# read training features
print("reading features")
train_name = 'features_split{}_all_nmfcc20.npy'
X_train = np.load(train_name.format(train_split))


# Read training labels
print("reading labels")
df = pd.read_csv('./train.csv')
y_train = df['Genre'].tolist()
y_train = y_train[:800]
y_train = np.repeat(y_train, train_split)

print("reading test features")
test_name = 'features_test_split{}_all_nmfcc20.npy'
X_test = np.load(test_name.format(test_split))



# normalize the data and encode the labels
encoder = LabelEncoder()
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
y_train = encoder.fit_transform(y_train)


net = RandomForestClassifier(n_estimators=100, random_state=42)


print("\nTraining")
net.fit(X_train, y_train)
print("Done ")



# Evaluate
acc_train = net.score(X_train, y_train)
print("\nAccuracy on train = %0.4f " % acc_train)



y_predict = net.predict(X_test)
y_predict = encoder.inverse_transform(y_predict)
utils.write_csv(y_predict=y_predict, fname='svm_split5_fs')