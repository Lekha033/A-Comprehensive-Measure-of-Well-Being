import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Read the dataset
data = pd.read_csv("hdi.csv")

# Input columns
X = data[["Life Expectancy", "Mean Years of Schooling", "GNI Per Capita"]]

# Output column
y = data["HDI Score"]

# Create the model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Save the model
with open("model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model Trained Successfully!")