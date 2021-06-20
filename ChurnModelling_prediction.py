import numpy as np 
import pandas as pd
import tensorflow as tf

df = pd.read_csv('../input/churn-predictions-personal/Churn_Predictions.csv')
y = df.iloc[:,13].values
X = df.iloc[:, 3:13].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
LabelEncoder_X_1 = LabelEncoder()
X[:,1] = LabelEncoder_X_1.fit_transform(X[:,1])

LabelEncoder_X_2 = LabelEncoder()
X[:,2] = LabelEncoder_X_2.fit_transform(X[:,2])

one_hot = OneHotEncoder()
one_hot_encoder = ColumnTransformer(transformers=[('one_hot', one_hot,[1])],remainder='passthrough')
X = one_hot_encoder.fit_transform(X)



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)


# Couche d'entrer
model = tf.keras.Sequential(tf.keras.layers.Dense(units=512,
                                  kernel_initializer = "uniform",
                                  activation = "relu",
                                  input_dim = 12))
# On ajoute une couche caché
model.add(tf.keras.layers.Dense(units=512,
                                kernel_initializer = "uniform",
                                activation = "relu"))
model.add(tf.keras.layers.Dense(units=1,
                                  kernel_initializer = "uniform",
                                  activation = "sigmoid"))



model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

model.fit(X_train, y_train, batch_size=25, epochs=500)

# Predicting the Test set results
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

print(cm)

# Evaluate
def build_classifier():
    classifier = Sequential()
    classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dense(units = 512, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = "adam", loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

# Evaluate
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 25, epochs = 500)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, 
                             cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()


