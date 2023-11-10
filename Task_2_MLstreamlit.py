import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
zoo = fetch_ucirepo(id=111) 

# data (as pandas dataframes) 
X = zoo.data.features 
y = zoo.data.targets 

# Streamlit UI
st.title("Zoo Dataset Classification App")

# Allow the user to choose the classifier
classifier_option = st.selectbox("Choose a Classifier", ["Random Forest", "Gradient Boosting", "Logistic Regression"])

# Allow the user to set parameters
if classifier_option == "Random Forest":
    n_estimators = st.slider("Number of Trees:", min_value=1, max_value=100, value=10)

elif classifier_option == "Gradient Boosting":
    learning_rate = st.slider("Learning Rate:", min_value=0.01, max_value=1.0, value=0.1)
    n_estimators = st.slider("Number of Trees:", min_value=1, max_value=100, value=10)

elif classifier_option == "Logistic Regression":
    max_iter = st.slider("Maximum Iterations:", min_value=100, max_value=5000, value=2000)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)  # 70% training and 30% test

# Make sure that y_train is a 1d array because .fit() method expects a 1d array
y_train_flat = np.ravel(y_train)

# Train and evaluate the selected classifier
if classifier_option == "Random Forest":
    rf_classifier = RandomForestClassifier(n_estimators=n_estimators)
    rf_classifier.fit(X_train, y_train_flat)
    y_pred = rf_classifier.predict(X_test)

elif classifier_option == "Gradient Boosting":
    gb_classifier = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators)
    gb_classifier.fit(X_train, y_train_flat)
    y_pred = gb_classifier.predict(X_test)

elif classifier_option == "Logistic Regression":
    logreg_classifier = LogisticRegression(max_iter=max_iter)
    logreg_classifier.fit(X_train, y_train_flat)
    y_pred = logreg_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
st.write(f"{classifier_option} Accuracy: {accuracy}")
