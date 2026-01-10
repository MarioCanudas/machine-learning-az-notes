from keras import Sequential, layers
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

df = pd.read_csv("Churn_Modelling.csv")

X = df.iloc[:, 3:-1].values
y = df.iloc[:, -1].values

# Make the Geography column an dummy variable
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])

ct = ColumnTransformer(
    transformers=[("encoder", OneHotEncoder(), [1])], remainder="passthrough"
)
X = np.array(ct.fit_transform(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

ann = Sequential()
ann.add(layers.Dense(units=6, activation="relu"))
ann.add(layers.Dense(units=6, activation="relu"))
ann.add(layers.Dense(units=1, activation="sigmoid"))

ann.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

ann.fit(X_train, y_train, batch_size=32, epochs=100)

y_pred = ann.predict(X_test)
y_pred = y_pred > 0.5

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

print(cm)
