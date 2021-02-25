import inline
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20, 10
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv("NSE-TATAGLOBAL.csv")
print(df)
print(df.shape)

df["Date"] = pd.to_datetime(df.Date, format="%Y-%m-%d")
df.index = df['Date']

plt.figure(figsize = (16, 8))
plt.plot(df["Close"], label='Close Price history')

#plt.show()

data = df.sort_index(ascending=True,axis=0)
new_dataset = pd.DataFrame(index=range(0, len(df)), columns=['Date', 'Close'])

for i in range(0, len(data)):
    new_dataset["Date"][i] = data['Date'][i]
    new_dataset["Close"][i] = data["Close"][i]

new_dataset.index = new_dataset.Date
new_dataset.drop("Date", axis=1, inplace=True)

# 5. Normalize the new filtered dataset:
final_dataset = new_dataset.values
nrow, nccol = final_dataset.shape
spl_nrow = int(0.65*nrow)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(final_dataset)

x_train_data, y_train_data = [], []

for i in range(60, spl_nrow):
    x_train_data.append(scaled_data[i - 60:i, 0])
    y_train_data.append(scaled_data[i, 0])

x_train_data, y_train_data = np.array(x_train_data), np.array(y_train_data)
x_train_data = np.reshape(x_train_data, (x_train_data.shape[0], x_train_data.shape[1], 1))

# 6. Build and train the LSTM model:
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train_data.shape[1], 1)))
lstm_model.add(LSTM(units=50))
lstm_model.add(Dense(1))

inputs_data = new_dataset[len(new_dataset) - (len(final_dataset)-spl_nrow) - 60:].values
inputs_data = inputs_data.reshape(-1, 1)
inputs_data = scaler.transform(inputs_data)

lstm_model.compile(loss='mean_squared_error', optimizer='adam')
lstm_model.fit(x_train_data, y_train_data, epochs=1, batch_size=1, verbose=2)

# 7. Take a sample of a dataset to make stock price predictions using the LSTM model:
X_test = []
for i in range(60, inputs_data.shape[0]):
    X_test.append(inputs_data[i-60:i, 0])

X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_closing_price = lstm_model.predict(X_test)
predicted_closing_price = scaler.inverse_transform(predicted_closing_price)

# 8. Save the LSTM model:
lstm_model.save("saved_model.h5")

# 9. Visualize the predicted stock costs with actual stock costs:
train_data = new_dataset[:spl_nrow]
valid_data = new_dataset[spl_nrow:]

valid_data['Predictions'] = predicted_closing_price.copy()

plt.plot(train_data['Close'])
plt.plot(valid_data[['Close', 'Predictions']])
plt.show()
print(1)
