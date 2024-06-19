

```python
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the Boston Housing dataset
boston = load_boston()
data = pd.DataFrame(boston.data, columns=boston.feature_names)
data['PRICE'] = boston.target

# Define features and target variable
X = data[['RM', 'LSTAT', 'PTRATIO']]  # Features: average number of rooms, % lower status of the population, pupil-teacher ratio
y = data['PRICE']  # Target variable: housing prices

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
```

In this code:
- We load the Boston Housing dataset and select specific features (average number of rooms, % lower status of the population, pupil-teacher ratio) and the target variable (housing prices).
- Split the data into training and testing sets.
- Train a linear regression model on the training data.
- Make predictions using the test data.
- Evaluate the model by calculating the Mean Squared Error (MSE).

You can run this code in your Python environment to perform linear regression on housing prices based on the specified features. Let me know if you need further assistance or explanations!
