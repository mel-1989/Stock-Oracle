from datetime import datetime, timedelta
from tensorflow import keras
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import os
import yfinance as yf
import pickle

def prepare_data(dataslice,symbol):
    window_size = 7
    
    x = []

    encodings = get_encodings()

    price_data = np.column_stack((
        dataslice['Open'],
        dataslice['High'],
        dataslice['Low'],
        dataslice['Close']
    ))
    symbol_encoding = get_encodings()[symbol.upper()]
    symbol_features = np.tile(symbol_encoding, (window_size, 1))
    features = np.hstack([price_data, symbol_features])
    x.append(features)

    x = np.array(x)
    x = x.reshape(1, 7, 507)

    return x

def gather_predicts(symbol, forward):
    try:
        print(f"Starting gather_predicts for {symbol} with forward={forward}")
        
        ticker = yf.Ticker(symbol)
        tickerdata = ticker.history(start=calculate_weekdays_earlier(50))
        print(f"Retrieved {len(tickerdata)} rows of ticker data")
        
        tickerdata = tickerdata.reset_index()
        
        # Load model
        print(f"Loading model for {symbol}")
        match symbol:
            case 'V':
                MODEL = load_model("model_(\'VUNH\', [\'V\', \'UNH\'])auto.keras")
            case 'UNH':
                MODEL = load_model("model_(\'VUNH\', [\'V\', \'UNH\'])auto.keras")
            case 'GOOG':
                MODEL = load_model('model_(\'GOOGMETA\', [\'GOOG\', \'META\'])auto.keras')
            case 'META':
                MODEL = load_model('model_(\'GOOGMETA\', [\'GOOG\', \'META\'])auto.keras')
            case 'AMZN': 
                MODEL = load_model('model_(\'AMZNTSLA\', [\'AMZN\', \'TSLA\'])auto.keras')
            case 'TSLA': 
                MODEL = load_model('model_(\'AMZNTSLA\', [\'AMZN\', \'TSLA\'])auto.keras')
            case _:
                MODEL = load_model("aaplnvda.keras")
        
        formatted_predictions = []
        
        for x in range(forward):
            target = len(tickerdata)-8+x 
            start = target-8
            
            if start < 0 or target <= 0:
                print(f"Skipping iteration {x} due to invalid indices: start={start}, target={target}")
                continue
                
            print(f"Processing window {x}: start={start}, target={target}")
            data_slice = tickerdata[start:target-1]
            print(f"Data slice shape: {data_slice.shape}")
            
            prepared_data = prepare_data(data_slice, symbol)
            print(f"Prepared data shape: {prepared_data.shape}")
            
            raw_prediction = MODEL.predict(prepared_data, verbose=0)
            print(f"Raw prediction shape: {raw_prediction.shape}")
            
            formatted_pred = {
                "date": tickerdata.iloc[start+8]["Date"].strftime('%Y-%m-%d') if start+8 < len(tickerdata) else None,
                "actual": float(tickerdata.iloc[target]["Close"]) if target < len(tickerdata) else -1,
                "predicted": float(raw_prediction[0][0])
            }
            formatted_predictions.append(formatted_pred)
            
        print(f"Returning {len(formatted_predictions)} predictions")
        return formatted_predictions
        
    except Exception as e:
        import traceback
        print(f"Error in gather_predicts: {str(e)}")
        print(f"Full traceback: {traceback.format_exc()}")
        raise

def test(symbol):
    x = []

    window_size = 7
    ticker = yf.Ticker(symbol)
    tickerdata  = ticker.history(start = calculate_weekdays_earlier(315))

    results = []
    expected = [tickerdata.iloc[i]['Close'] for i in range(7, len(tickerdata), 7)]

    num_windows = min(int(315/window_size), len(tickerdata)//window_size)
    for n in range(num_windows):
        match symbol:
            case 'V':
                MODEL = load_model("model_(\'VUNH\', [\'V\', \'UNH\'])auto.keras")
            case 'UNH':
                MODEL = load_model("model_(\'VUNH\', [\'V\', \'UNH\'])auto.keras")
            case 'GOOG':
                MODEL = load_model('model_(\'GOOGMETA\', [\'GOOG\', \'META\'])auto.keras')
            case 'META':
                MODEL = load_model('model_(\'GOOGMETA\', [\'GOOG\', \'META\'])auto.keras')
            case _:
                MODEL = load_model("aaplnvda.keras")
        
        start = n*window_size
        end = (n+1)*window_size
        y_hat = MODEL.predict(prepare_data(tickerdata[start:end],symbol))
        results.append(y_hat)


    difference = [x-y for x,y in zip(results, expected)]
    average = sum(difference) / len(difference)

    data = {
        "results" :  results,
        "expected": expected,
        "difference" : difference
    }
    
    eval = pd.DataFrame(data=data)
    print("symbol:" + symbol)
    print(eval)
    print("average" +str(average))
    return [eval, average, symbol]


    
def calculate_weekdays_earlier(days):
    # Get the current date
    current_date = datetime.now()

    # Initialize variables
    weekdays_count = 0
    delta_days = 0

    # Iterate to count back days weekdays
    while weekdays_count < days + 1:
        delta_days += 1
        new_date = current_date - timedelta(days=delta_days)
        # Check if it's a weekday (Monday=0, Sunday=6)
        if new_date.weekday() < 5:  # 0-4 are weekdays
            weekdays_count += 1
            
    # Return the calculated date in 'YYYY-MM-DD' format
    return new_date.strftime('%Y-%m-%d')


def get_encodings():
    # Load the dictionary
    with open('symbol_to_onehot_dict.pkl', 'rb') as f:
        loaded_symbol_to_onehot_dict = pickle.load(f)
    
    return loaded_symbol_to_onehot_dict

if __name__ == '__main__':
    test('V')
    test("UNH")
    test("GOOG")
    test('META')
    test("BRK.B")
    test("JPM")
    test("AMZN")
    test("TSLA")
    test("AAPL")
    test("NVDA")

    # workbook = xlsxwriter.Workbook("Evaluation_" + str(datetime.now))
    # worksheet = workbook.add_worksheet("firstSheet")

    # worksheet.write(0,0, "#")
    # worksheet.write(0,1, "V")
    # worksheet.write(0,2, "UNH")
    # worksheet.write(0,3, "GOOG")
    # worksheet.write(0,4, "META")
    # worksheet.write(0,5, "BRK.B")
    # worksheet.write(0,6, "JPM")
    # worksheet.write(0,7, "AMZN")
    # worksheet.write(0,8, "TSLA")
    # worksheet.write(0,9, "AAPL")
    # worksheet.write(0,10, "NVDA")

    # worksheet.write(0,0, "#")
    # worksheet.write(0,1, "V")
    # worksheet.write(0,2, "UNH")
    # worksheet.write(0,3, "GOOG")
    # worksheet.write(0,4, "META")
    # worksheet.write(0,5, "BRK.B")
    # worksheet.write(0,6, "JPM")
    # worksheet.write(0,7, "AMZN")
    # worksheet.write(0,8, "TSLA")
    # worksheet.write(0,9, "AAPL")
    # worksheet.write(0,10, "NVDA")

