import yfinance as yf
import numpy as np, pandas as pd, matplotlib.pyplot as plt

from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD


# Import Dataset
df = yf.download("ETH-USD", start = "2019-01-01", end = datetime.today(), interval= "1d")

# Data Preparing

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1,1))

predict_days = 180

x_train = []
y_train = []

for i in range(predict_days, len(scaled_data)):
    x_train.append(scaled_data[i-predict_days:i, 0])
    y_train.append(scaled_data[i, 0])


x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# Neural Network Stage

model = Sequential()
model.add(LSTM(units = 50, return_sequences = True, input_shape = (x_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units = 50, return_sequences = True))
model.add(Dropout(0.2))
model.add(LSTM(units = 50))
model.add(Dropout(0.2))

model.add(Dense(units = 1))
sgd = SGD(learning_rate=0.1, momentum=0.9)

model.compile(optimizer = sgd, loss = "mean_squared_error")
model.fit(x_train, y_train, epochs = 25, batch_size = 32)


# Model Testing

test_start = datetime(2020,1,1) + timedelta(days=-predict_days)
test_end = datetime.now()

test_data = yf.download("ETH-USD", start = test_start, end = test_end, interval= "1d")
real_prices = test_data["Close"].values

full_data = pd.concat((df["Close"], test_data["Close"]))

model_inputs = full_data[len(full_data) - len(test_data) - predict_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.fit_transform(model_inputs)

x_test = []

for i in range(predict_days, len(model_inputs)):
    x_test.append(model_inputs[i-predict_days:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

prediction_prices = model.predict(x_test)
prediction_prices = scaler.inverse_transform(prediction_prices)

plt.plot(real_prices, color = "red", label = "Actual Prices")
plt.plot(prediction_prices, color = "blue", label = "Prediction Prices")
plt.title("ETH Prediction")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()


# Future Prediction

real_data = [model_inputs[len(model_inputs) + 1 - predict_days: len(model_inputs) + 1, 0 ]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
print("Next Day ETH Price Prediction", prediction)