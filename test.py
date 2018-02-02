# -- encoding: utf-8 --

import os
import json

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report,confusion_matrix

from sklearn.naive_bayes import MultinomialNB

datalist = []
targetlist = []

for filename in os.listdir("data"):
    with open("data/" + filename) as f:
        try:
            file_data = json.loads(f.read().encode('utf-8').strip())
            messages = ""
            for item in file_data['chat_log']:
                messages += " " + item['message']
            datalist.append(messages)
            targetlist.append(file_data['allied_report_count'])
        except:
            continue

X = datalist
Y = targetlist

X_train, X_test, y_train, y_test = train_test_split(X, Y)


scaler = StandardScaler()
# Fit only to the training data
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(30,30,30))

mlp.fit(X_train,y_train)

predictions = mlp.predict(X_test)

print(confusion_matrix(y_test,predictions))

print(classification_report(y_test,predictions))