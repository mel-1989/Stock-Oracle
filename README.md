# Stock-Oracle:
A machine learning framework for predicting stock market movements using historical price data. Future plans to leverage technical indicators and sentiment analysis from financial news. Disclaimer: I've made this project with the goal of getting my feet wet with ML modelling in a somewhat practical space. This is by no means a fully developed project or anything meant to be deployed or used in any real world context. Hopefully, this first test will lead to greater project in the future, but for now this is only a educational endeavor. 

My conclusion: the methods used here alone are unsatisfactory for price prediction, even putting aside wildly inaccurate estimates (some tickers resulted in consistently poor outputs, likely a result of historical events distorting the data's continuity in atypical ways, such as stock splits), historical data alone is certainly not enough to attempt medium term valuation (daily and weekly shifts). I have not tried with shorter time scales due to computational limitations.

Experiments for future projects:
- greater selection of data types (fundamentals, multiples, technical indicators, macro, sentiment(?))
- greater specificity in prediction (industrial groupings, boutique data selection for a single stock, etc)
- Different time scales (long term (present value) estimates, ultra short term HFT)
- Different output methodology (buy/sell signal instead of price estimate, more general rating system)
- Develop a more solid model architecture design (bounced back and forth with a few different models, need some thorough research on design)

# Overview:
Stock Oracle combines multiple ML models including LSTM networks and gradient boosting to forecast stock prices across different time horizons. The system processes real-time market data to generate trading signals and price predictions.
#Key Features:
- Multi-model ensemble approach combining deep learning and traditional ML algorithms
- Real-time processing of market data and technical indicators
- Customizable prediction horizons
- Comprehensive testing framework with performance metrics
- locally hosted web visualization system with customization options

# Coming Soon:
- Natural language processing of financial news and social media sentiment
- inclusion of technical indicators to increase accuracy

# Getting Started
Stock Oracle is designed to get you up and running with minimal setup. Follow these steps to start predicting market movements:
Prerequisites

- Python 3.10
- Node.js 18+
- Git

# 1. Clone the Repository
  
```
git clone https://github.com/mel-1989/Stock-Oracle.git
pip install -r requirements.txt
```

# 2. Train the Models
  Edit Auto.py to train models for the stocks you want to predict. To do this, just find the "stock_pairs"       dictionary in the file, edit the elements to include the stock or stock pairs you want to focus on. 
  

  Once the stocks are specified, use the following command:
  
```
 python auto.py
```
 
  This will:
  - Download historical data for the specified symbols
  - Process data automatically and begin the to train the ensemble of ML models
  - optimize the model structure iteratively
  - Save the trained models
    

The training process typically takes 15-30 minutes per symbol on a modern CPU. GPU acceleration is supported and will significantly reduce training time.

# 3. Launch the Visualization Dashboard
Start the web interface to explore predictions and model insights:

launch the backend first:

```
 python api.py
```

Then in a different terminal activate the react frontend:

```
  npm install #on first install
  npm start
```
  
Navigate to http://localhost:3000 to access the dashboard. You'll see real-time predictions, model performance metrics, and interactive visualizations of market trends.
That's it! 

# Disclaimer
This project is for educational purposes only. Stock market prediction is inherently uncertain and past performance does not guarantee future results. Always conduct thorough research and consider consulting financial advisors before making investment decisions.
