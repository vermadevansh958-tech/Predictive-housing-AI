import numpy as ko
import pandas as po
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

ko.random.seed(56)
num_houses = 200
sizes = ko.random.randint(600, 3600, size=num_houses)

bedrooms = []
for size in sizes:
    if size < 1200:
        bedrooms.append(ko.random.choice([1, 2]))
    elif size < 2200:
        bedrooms.append(ko.random.choice([2, 3]))
    else:
        bedrooms.append(ko.random.choice([4, 5]))
bedrooms = ko.array(bedrooms)

ages = ko.random.randint(1, 25, size=num_houses)
base_prices = 20 + (sizes * 0.05) + (bedrooms * 10) - (ages * 0.4)
market_noise = ko.random.normal(0, 5, size=num_houses)
prices_lakhs = ko.round(base_prices + market_noise, 2)

data = {
    'Size_SqFt': sizes,
    'Bedrooms': bedrooms,
    'Age_Years': ages,
    'Price_INR_Lakhs': prices_lakhs
}
df = po.DataFrame(data)

X = df[['Size_SqFt', 'Bedrooms', 'Age_Years']]
y = df['Price_INR_Lakhs']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=56)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

custom_house = po.DataFrame([[2200, 4, 3]], columns=['Size_SqFt', 'Bedrooms', 'Age_Years'])
predicted_price = model.predict(custom_house)
print(f"Predicted Price: {predicted_price[0]:.2f}")
