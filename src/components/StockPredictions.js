import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Flame, TrendingUp } from 'lucide-react';

const StockPredictions = () => {
  const [selectedStock, setSelectedStock] = useState('AAPL');
  useEffect(() => {
    if (selectedStock) {
      generatePredictions();
    }
  }, [selectedStock]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [stats, setStats] = useState({
    accuracy: null,
    confidence: null,
    nextPrediction: null
  });

  const stockOptions = [
    { value: 'AAPL', label: 'Apple Inc.' },
    { value: 'NVDA', label: 'NVIDIA Corporation' },
    { value: 'V', label: 'VISA' },
    { value: 'UNH', label: 'United Health Care' },
    { value: 'BRK.B', label: 'Berkshire Hathaway' },
    { value: 'GOOG', label: 'Google' },
    { value: 'META', label: 'Meta' },
    { value: 'JPM', label: 'JP Morgan' },
    { value: 'TSLA', label: 'Tesla' }
  ];

  const generatePredictions = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbol: selectedStock,
          forward: 1, //develop this further later
          window_size: 7  // Adjust based on your model's requirements
        }),
      });
      const data = await response.json();
      
      if (!data.success) {
        throw new Error(data.error || 'Failed to generate predictions');
      }
      console.log("skibidi biden2")
      console.log(data);
      setPredictions(data.predictions);
      
      // Calculate statistics from predictions
      if (data.predictions.length > 0) {
        const latestPrediction = data.predictions[data.predictions.length - 1];
        
        // Calculate accuracy (mean absolute percentage error)
        const mape = data.predictions.reduce((sum, point) => {
          return sum + Math.abs((point.actual - point.predicted) / point.actual);
        }, 0) / data.predictions.length;
        
        const accuracy = ((1 - mape) * 100).toFixed(1);
        
        // Determine confidence based on prediction variance
        const predictionErrors = data.predictions.map(p => Math.abs(p.actual - p.predicted));
        const avgError = predictionErrors.reduce((a, b) => a + b, 0) / predictionErrors.length;
        const confidence = avgError < 5 ? "High" : avgError < 10 ? "Medium" : "Low";
        
        setStats({
          accuracy: accuracy,
          confidence: confidence,
          nextPrediction: latestPrediction.predicted
        });
      }
    } catch (err) {
      setError(err.message);
      console.error('Error generating predictions:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 p-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="flex items-center gap-3 mb-8">
          <Flame className="text-orange-500" size={32} />
          <h1 className="text-4xl font-bold text-orange-500">Oracle of Wall Street</h1>
        </div>

        {/* Controls */}
        <div className="bg-gray-800 rounded-lg p-6 mb-8">
          <div className="flex flex-col md:flex-row gap-4 items-start md:items-center">
            <select
              value={selectedStock}
              onChange={(e) => setSelectedStock(e.target.value)}
              className="bg-gray-700 text-orange-100 rounded-lg px-4 py-2 border border-orange-500/30 focus:border-orange-500 focus:ring-2 focus:ring-orange-500/20 outline-none"
              disabled={loading}
            >
              {stockOptions.map(option => (
                <option key={option.value} value={option.value}>
                  {option.label}
                </option>
              ))}
            </select>
            
            <button
              onClick={generatePredictions}
              disabled={loading}
              className={`flex items-center gap-2 bg-gradient-to-r from-orange-600 to-red-600 text-white px-6 py-2 rounded-lg transition-colors ${
                loading ? 'opacity-50 cursor-not-allowed' : 'hover:from-orange-500 hover:to-red-500'
              }`}
            >
              <TrendingUp size={20} />
              {loading ? 'Generating...' : 'Generate Predictions'}
            </button>
          </div>
          
          {error && (
            <div className="mt-4 p-4 bg-red-900/50 border border-red-500/50 rounded-lg text-red-200">
              {error}
            </div>
          )}
        </div>

        {/* Chart */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-2xl font-semibold text-orange-100 mb-6">Price Predictions</h2>
          <div className="h-96">
            {predictions.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={predictions}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#D1D5DB"
                  />
                  <YAxis 
                    stroke="#D1D5DB"
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#1F2937',
                      border: '1px solid #374151',
                      borderRadius: '0.5rem',
                    }}
                    labelStyle={{ color: '#D1D5DB' }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="actual" 
                    stroke="#F97316" 
                    strokeWidth={2}
                    dot={{ fill: '#F97316' }}
                    name="Actual Price"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="predicted" 
                    stroke="#EA580C" 
                    strokeWidth={2}
                    dot={{ fill: '#EA580C' }}
                    name="Predicted Price"
                    strokeDasharray="5 5"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-gray-400">
                Generate predictions to see the chart
              </div>
            )}
          </div>
        </div>

        {/* Stats */}
        {predictions.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
            <div className="bg-gray-800 p-6 rounded-lg border border-orange-500/20">
              <h3 className="text-orange-500 text-lg font-medium mb-2">Accuracy</h3>
              <p className="text-3xl font-bold text-white">{stats.accuracy}%</p>
            </div>
            <div className="bg-gray-800 p-6 rounded-lg border border-orange-500/20">
              <h3 className="text-orange-500 text-lg font-medium mb-2">Confidence</h3>
              <p className="text-3xl font-bold text-white">{stats.confidence}</p>
            </div>
            <div className="bg-gray-800 p-6 rounded-lg border border-orange-500/20">
              <h3 className="text-orange-500 text-lg font-medium mb-2">Next Prediction</h3>
              <p className="text-3xl font-bold text-white">
                ${stats.nextPrediction?.toFixed(2)}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StockPredictions;