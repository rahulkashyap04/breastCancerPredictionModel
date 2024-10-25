import numpy as np
import pandas as pd

# Load the dataset
breast = pd.read_csv('breastdataset.csv')

# Display the first 20 rows
print(breast.head(20))

# Check the shape and info of the dataframe
print(breast.shape)
print(breast.info())

# Count the occurrences of each diagnosis
print(breast['diagnosis'].value_counts())

# Drop the unnecessary column
breast.drop('Unnamed: 32', axis=1, inplace=True)

# Check the shape again
print(breast.shape)

# Map the diagnosis to numerical values
breast['diagnosis'] = breast['diagnosis'].map({"M": 1, "B": 0})

# Check the updated counts
print(breast['diagnosis'].value_counts())

# Separate features and target variable
x = breast.drop('diagnosis', axis=1)
y = breast['diagnosis']

print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print(x_train.shape)
print(x_test.shape)

from sklearn.preprocessing import StandardScaler

# Standardize the data
sc = StandardScaler()
sc.fit(x_train)

x_train = sc.transform(x_train)
x_test = sc.transform(x_test)

print(x_train)
print(x_test)

from sklearn.linear_model import LogisticRegression

# Create and train the logistic regression model
lg = LogisticRegression()
lg.fit(x_train, y_train)

# Make predictions
y_pred = lg.predict(x_test)

print(y_pred)

from sklearn.metrics import accuracy_score

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

# Sample input for prediction
input_text = (-0.23720599, 1.61168786, 0.69578703, 1.56893649, 1.7081298,
              0.18450055, -0.01753996, 0.74327887, 1.22835748, -0.83398079,
              -1.22285922, 1.25927075, -0.34313266, 1.45328482, 1.53289495,
              -0.16213492, -0.36077377, 0.03383119, 0.5274313, -0.85918669,
              -0.6549607, 2.30648136, 0.88212609, 2.39057394, 2.68233225,
              0.85300607, 0.40207085, 1.25547696, 1.90972637, -0.21721319,
              -0.43289919)

np_df = np.asarray(input_text)
pred = lg.predict(np_df.reshape(1, -1))

# Output the prediction result
if pred[0] == 1:
    print("Cancers")
else:
    print("Not cancers")

# Save the model
import pickle
pickle.dump(lg, open('model.pkl', 'wb'))
