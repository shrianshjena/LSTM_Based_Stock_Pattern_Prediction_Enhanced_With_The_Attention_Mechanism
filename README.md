# LSTM-Based Stock Pattern Prediction Enhanced With The Attention Mechanism

## Overview

This project investigates the application of Long Short-Term Memory (LSTM) networks, enhanced with an attention mechanism, for predicting stock prices. The primary objective is to develop a model that can accurately forecast future stock prices by leveraging historical data, thus aiding investors and analysts in making informed decisions.

### Why Use LSTM?

LSTM is a type of Recurrent Neural Network (RNN) that is well-suited for time series forecasting tasks due to its ability to remember information for long periods. Unlike standard feedforward neural networks, LSTMs have feedback connections, making them capable of processing entire sequences of data, making them ideal for stock price prediction where historical data plays a crucial role in forecasting future prices.

### Why Use Attention Mechanism?

The attention mechanism, initially developed for sequence-to-sequence models in natural language processing, allows the model to focus on the most relevant parts of the input sequence when making predictions. By integrating attention with LSTM, our model can dynamically weight the importance of different time steps in the input sequence, leading to more accurate and interpretable predictions.

### Project Goals

1. **Accurate Predictions**: Utilize LSTM and attention mechanisms to make precise short-term stock price predictions.
2. **Interpretability**: Provide insights into which parts of the input data are most influential in the model's predictions.
3. **Scalability**: Create a framework that can be extended to other stocks and potentially other time series forecasting tasks.

### Key Steps in the Project

1. **Data Collection and Preprocessing**: The project starts with collecting historical stock price data, including features such as `Date`, `Open`, `High`, `Low`, `Close`, and `Volume`. This data is then preprocessed by normalizing the values to fit within a specific range, which helps in speeding up the training process and improving model performance.

2. **Model Building**: The core of the project is building the LSTM model with an integrated attention mechanism. The LSTM network captures the temporal dependencies in the data, while the attention mechanism helps the model to focus on the most relevant time steps in the input sequence.

3. **Model Training**: The model is trained on the historical stock price data. During training, the model learns to predict future prices by minimizing the error between the predicted and actual prices.

4. **Prediction and Evaluation**: After training, the model is used to predict future stock prices. The performance of the model is evaluated by comparing the predicted prices with actual prices. Visualization tools are used to plot the results, providing a clear view of the model's accuracy.

5. **User Interaction**: The notebook allows users to input a specific date and predict stock prices for the subsequent days. This interactive feature makes the tool practical for real-world use.

### Benefits of the Approach

- **Enhanced Predictive Power**: By combining LSTM with an attention mechanism, the model benefits from both temporal understanding and selective focus on important data points, resulting in more accurate predictions.
- **Practical Application**: Investors and analysts can use the model to get insights into future stock prices, aiding in decision-making.
- **Extensibility**: The framework can be extended beyond stock prices to any time series data, making it a versatile tool for various forecasting tasks.

This project not only showcases the power of modern neural networks in time series forecasting but also provides a practical tool that can be used and extended by data scientists and financial analysts.

## Features

- **Data Preprocessing**: Collection and preprocessing of historical stock data.
- **Model Architecture**: Implementation of an LSTM model with an attention mechanism to enhance prediction accuracy.
- **Training**: Model training using historical stock price data.
- **Prediction**: Predicting future stock prices based on the trained model.
- **Visualization**: Plotting and visualizing the results of the predictions.

## Key Components

### Data Preprocessing

- **Loading Data**: Historical stock data is loaded.
- **Normalization**: Data is normalized to fit within a specific range.
- **Sequence Generation**: Data is structured into sequences for LSTM input.

### Model Architecture

- **LSTM Network**: A deep learning model designed for sequence prediction.
- **Attention Mechanism**: Enhances the LSTM model by focusing on important parts of the input sequence.

### Training and Prediction

- **Model Training**: The LSTM model is trained on historical stock data.
- **Stock Price Prediction**: The trained model predicts future stock prices.

### Visualization

- **Plotting**: Visualizing the predicted vs. actual stock prices.

## Results

The model was trained on a dataset of 756 samples and tested on 190 samples. Key performance metrics include:
- **Test Loss**: 0.0053
- **Mean Absolute Error (MAE)**: 0.0683
- **Root Mean Square Error (RMSE)**: 0.0729

The model successfully predicts stock prices for the next four days with a reasonable accuracy. The attention mechanism has shown improvements in focusing on relevant parts of the data, enhancing the prediction performance.

Visualizations include:
- Historical stock prices with trading volume.
- Predicted stock prices for the next four days.
- Comparison of actual stock prices for the past 60 days with predicted prices for the next four days.

## Discussion

### Analysis of Results

The LSTM model with the attention mechanism demonstrates high accuracy and reliability in predicting stock prices. The attention mechanism enhances the model's ability to focus on relevant parts of the input sequence, improving performance.

### Comparison with Other Methods

Compared to traditional methods and basic machine learning models, the LSTM network with attention mechanism outperforms in terms of accuracy and handling complex dependencies in stock price data.

### Limitations

- The model's performance depends on the quality and quantity of historical data.
- Primarily designed for short-term predictions.
- Sensitive to sudden market events and requires significant computational resources.

## Conclusion

This project demonstrates the use of LSTM networks enhanced with an attention mechanism for stock price prediction. It provides a robust framework for time series forecasting and can be extended to other types of financial data.
