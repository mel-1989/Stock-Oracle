import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import TimeSeriesSplit, train_test_split
import optuna
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class StockPairModeler:
    def __init__(self, window_size, n_features):
        self.window_size = window_size
        self.n_features = n_features
        self.best_trial = None
        self.best_model = None
        
    def create_model(self, trial):
        """Create model with hyperparameters suggested by Optuna"""
        model = Sequential()
        
        # Number of LSTM layers (1-3)
        n_layers = trial.suggest_int('n_layers', 1, 3)
        
        # First LSTM layer
        units = trial.suggest_int('lstm_units_1', 32, 256)
        dropout = trial.suggest_float('dropout_1', 0.2, 0.8)
        model.add(LSTM(units, 
                      input_shape=(self.window_size, self.n_features),
                      return_sequences=True if n_layers > 1 else False,
                      recurrent_dropout=dropout))
        
        # Additional LSTM layers
        for i in range(2, n_layers + 1):
            units = trial.suggest_int(f'lstm_units_{i}', 32, 256)
            dropout = trial.suggest_float(f'dropout_{i}', 0.2, 0.8)
            model.add(LSTM(units, 
                          return_sequences=True if i < n_layers else False,
                          recurrent_dropout=dropout))
        
        # Output layer
        model.add(Dense(1))
        
        # Compile model with suggested learning rate
        lr = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        model.compile(optimizer=Adam(learning_rate=lr),
                     loss='mean_squared_error')
        
        return model
    
    def objective(self, trial, X_train, y_train, X_val, y_val):
        """Optuna objective function for hyperparameter optimization"""
        # Create model with trial parameters
        model = self.create_model(trial)
        
        # Training parameters
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
        
        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=200,  # Maximum epochs
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0
        )
        
        return min(history.history['val_loss'])
    
    def optimize(self, X_train, y_train, n_trials=50):
        """Run hyperparameter optimization"""
        # Create time series split for validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Get last split for optimization
        for train_idx, val_idx in tscv.split(X_train):
            X_train_split = X_train[train_idx]
            y_train_split = y_train[train_idx]
            X_val_split = X_train[val_idx]
            y_val_split = y_train[val_idx]
        
        # Create and run study
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: self.objective(trial, 
                                                  X_train_split, 
                                                  y_train_split,
                                                  X_val_split,
                                                  y_val_split),
                      n_trials=n_trials)
        
        self.best_trial = study.best_trial
        
        # Train final model with best parameters
        self.best_model = self.create_model(study.best_trial)
        return study
    
    def train_final_model(self, X_train, y_train, X_test, y_test, model_name):
        """Train final model with best parameters and save it"""
        if self.best_model is None:
            raise ValueError("Must run optimize() first")
            
        checkpoint = ModelCheckpoint(
            f"{model_name}.keras",
            monitor='val_loss',
            save_best_only=True
        )
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6),
            checkpoint
        ]
        
        history = self.best_model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=300,
            batch_size=self.best_trial.params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        self.best_model.save(model_name+"auto"+".keras")
        
        return history


df2 = pd.read_csv('sp500_stocks.csv')

testdf = df2.fillna(-1)
testdf['Open'].isnull().unique()


df2.isna().any()
df2[df2['Open'].isna()]

"""#Baseline LSTM Model"""

window_size = 15
window_future = 1
def createdataset(testdf, window_size, window_future, pair_names):
  x = []
  y = []
  symbols = testdf['Symbol'].unique()

  encoder = OneHotEncoder(sparse_output = False)
  symbols_encoded = encoder.fit_transform(symbols.reshape(-1,1))


  symbol_to_onehot_dict = dict(zip(symbols, symbols_encoded))

  for sym in pair_names:
    sym_open = np.array(testdf[testdf['Symbol'] == sym]['Open'])
    sym_high = np.array(testdf[testdf['Symbol'] == sym]['High'])
    sym_low = np.array(testdf[testdf['Symbol'] == sym]['Low'])
    sym_close = np.array(testdf[testdf['Symbol'] == sym]['Close'])
    for i in range(len(sym_open) - window_future - window_size):
      sym_onehot = np.tile(symbol_to_onehot_dict[sym], (window_size, 1))
      # print(sym_onehot, "sym_onehot")
      # print(sym_onehot.shape, "sym_onehot")
      features = np.column_stack((sym_open[i:(i + window_size)],sym_high[i:(i + window_size)],sym_low[i:(i + window_size)], sym_close[i:(i + window_size)]))
      features = np.hstack([features, sym_onehot])
      # print(features.shape)
      # print(features)

      x.append(features)
      y.append(sym_close[i + window_size])



#  AMZN_open = np.array(testdf[testdf['Symbol'] == 'AMZN']['Open'])
 # AMZN_high = np.array(testdf[testdf['Symbol'] == 'AMZN']['High'])
#  AMZN_low = np.array(testdf[testdf['Symbol'] == 'AMZN']['Low'])
# AMZN_close = np.array(testdf[testdf['Symbol'] == 'AMZN']['Close'])


  return np.array(x), np.array(y)

# dictionary to store models for different pairs
stock_pair_models = {
    "aaplnvda": [],
    "amzntsla": [],
    "GOOGMETA": [],
    "BRKJPM": [],
    "VUNH": []
}
stock_pairs = {
    "aaplnvda": ["AAPL", "NVDA"],
    "amzntsla": ["AMZN", "TSLA"],
    "GOOGMETA": ["GOOG","META"],
    "BRK.BJPM": ["BRK.B","JPM"],
    "VUNH": ["V","UNH"]
}

# Usage example:
window_size = 7  # Your existing window size
n_features = 507

# For each stock pair
for pair_name in stock_pairs.items():
    print("PAIR NAME::!:!:!: " + str(pair_name[0]))
    x,y = createdataset(testdf, window_size, window_future, pair_names=pair_name[1])
    X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=42, shuffle=False)
    print(f"Training model for {pair_name}")
    
    modeler = StockPairModeler(window_size, n_features)
    study = modeler.optimize(X_train, y_train)
    history = modeler.train_final_model(X_train, y_train, X_test, y_test, f"model_{pair_name}")
    
    stock_pair_models[pair_name[0]] = [modeler, history]
    

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Model Loss')
plt.show()