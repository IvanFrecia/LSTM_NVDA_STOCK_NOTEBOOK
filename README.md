# Stock Price Prediction and Forecasting Notebook

This Jupyter Notebook provides a comprehensive solution for predicting and forecasting stock prices using a Long Short-Term Memory (LSTM) neural network model. The notebook is designed to download historical stock data, load a pre-trained model, make predictions for future stock prices, and optionally retrain the model with the latest data.

## Overview

This notebook performs the following key tasks:

1.  **Data Acquisition:** Downloads historical stock price data for a specified ticker (default: AMZN) from Yahoo Finance using the `yfinance` library.
2.  **Model Loading:** Loads a pre-trained LSTM model and a data scaler (joblib) from specified file paths.
3.  **Data Preprocessing:**
    *   Scales the downloaded data using the loaded scaler.
    *   Prepares the data into sequences suitable for LSTM input.
4.  **Prediction:**
    *   Takes a user-specified target date for forecasting.
    *   Calculates the number of business days between the last available trading day and the target date.
    *   Uses the loaded model to predict stock prices for the calculated forecast horizon.
    *   Inverts the scaling to return the predictions to the original price scale.
5.  **Visualization:** Generates a plot of the predicted stock prices over the forecast period.
6.  **Data Saving:** Saves the predicted stock prices and the corresponding dates to a CSV file and the plot to a PNG file.
7.  **Optional Retraining:**
    *   Provides an option to retrain the loaded model with the newly downloaded data.
    *   Creates new training sequences from the updated data.
    *   Fine-tunes the model using the new sequences.
    *   Saves the retrained model.
8. **Logging:** Implements a logging system to track the execution of the notebook, including key events, errors, and warnings.

## Model Architecture

The notebook utilizes a custom LSTM model defined as `StockLSTMModel`. The model architecture consists of:

*   **Two LSTM layers:** With a specified number of units (`lstm_units`, default: 100) and dropout layers for regularization.
*   **Two Dense layers:** The first with 25 units and ReLU activation, and the second with a number of units equal to the forecast horizon.
* **Custom Loss Function:** A custom weighted loss function (`custom_weighted_loss_v2`) is used to penalize errors in later time steps more heavily.

## Dataset

The notebook uses historical stock price data downloaded from Yahoo Finance. The primary feature used for prediction is the "Close" price. Optionally, other features can be added (commented out in the code), such as:

*   Volume
*   Daily Return
*   Simple Moving Averages (SMA)

## Libraries Used

*   **os:** For file and directory operations.
*   **datetime:** For date and time manipulation.
*   **logging:** For logging events and errors.
*   **numpy:** For numerical operations.
*   **pandas:** For data manipulation and analysis.
*   **yfinance:** For downloading stock data from Yahoo Finance.
*   **matplotlib.pyplot:** For data visualization.
*   **joblib:** For saving and loading the data scaler.
*   **tensorflow:** For building and training the LSTM model.
*   **tensorflow.keras:** For defining the model architecture and layers.

## How to Run

1.  **Clone the repository:** `git clone [repository_url]` (If applicable)
2.  **Install dependencies:** `pip install -r requirements.txt` (If you have a `requirements.txt` file)
    *   Ensure you have the required libraries installed. You can create a `requirements.txt` file using `pip freeze > requirements.txt` after installing the necessary packages.
3.  **Download the model and scaler:**
    *   Download the `amzn_base.keras` and `amzn_base_scaler.pkl` files and place them in the `/content/` directory.
4.  **Open the notebook:** Launch Jupyter Notebook or JupyterLab and open the notebook file (e.g., `amzn.ipynb`).
5.  **Run the cells:** Execute the notebook cells sequentially.
6.  **Set the Target Date:** Modify the `TARGET_DATE` variable to the desired forecast date.
7. **Retrain the model:** If you want to retrain the model, change the `RETRAIN_MODEL` variable to "y".
8. **Run the notebook:** Execute all the cells.

## Key Features

*   **Custom LSTM Model:** The notebook defines and uses a custom LSTM model with dropout and dense layers.
*   **Custom Weighted Loss:** A custom weighted loss function is implemented to improve the accuracy of predictions further into the future.
*   **Data Scaling:** The notebook uses a pre-trained scaler to normalize the data before feeding it to the model.
*   **Prediction and Visualization:** The notebook generates predictions and visualizes them in a plot.
*   **Data Saving:** The predictions and the plot are saved to CSV and PNG files, respectively.
*   **Optional Retraining:** The notebook provides an option to retrain the model with the latest data.
*   **Logging:** The notebook includes comprehensive logging to track the execution flow and any potential issues.

## Key Findings


*   **Example:** The model successfully predicts the general trend of the stock price over the forecast period.
*   **Example:** The retraining process improves the model's accuracy on the most recent data.
*   **Example:** The custom weighted loss function helps to reduce errors in the later part of the forecast horizon.

## Future Work


*   **Hyperparameter Tuning:** Optimize the model's hyperparameters (e.g., LSTM units, dropout rate, learning rate) to improve performance.
*   **Feature Engineering:** Explore additional features (e.g., technical indicators, sentiment analysis) to enhance the model's predictive power.
*   **Model Comparison:** Compare the performance of the LSTM model with other time-series forecasting models.
*   **Error Analysis:** Conduct a detailed error analysis to identify patterns in the model's mistakes.
*   **Real-time Prediction:** Implement a system for real-time stock price prediction.
*   **Automated Retraining:** Automate the retraining process to regularly update the model with new data.
* **Deploy the model:** Deploy the model as a web service.

## Contributing


Contributions to this project are welcome! Please feel free to submit pull requests or open issues to suggest improvements or report bugs.

## License

**MIT License**


