import time
import pandas as pd

data = pd.read_csv('biodeg.csv', sep=";", header=None)

#separate data
x = data.iloc[:,:-1]
y = data.iloc[:,-1]

#splitting data
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#fitting the model
start = time.time()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(solver="svd")
lda.fit(x_train, y_train)
end = time.time()
print("Training time:", end-start)

#prediction
start = time.time()
y_pred = lda.predict(x_test)
end = time.time()
print("Testing time:", end-start)

#total number of data points
total_test = len(x_test)
total_data = len(data)
print("Total number of data points:", total_data)
print("Total number of tested data points:", total_test)

#evaluation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#percentage of confusion matrix
total = cm.sum()
percentage = cm / total * 100
print("Percentage of Confusion Matrix: \n", percentage)
