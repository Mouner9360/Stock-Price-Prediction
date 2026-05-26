# Stock Price Prediction Using an Improved Transformer Model

> Predicting next-day stock prices with a custom transformer architecture featuring wavelet denoising, adaptive time windows, and convolution-enhanced self-attention -- achieving a 55.8% trade win rate and 17%+ annualized simulated profit on Verizon (VZ) stock.

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-ff6f00.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

---

## About

Predicting stock prices is notoriously difficult -- markets are volatile, irrational, and influenced by factors ranging from insider trading to media hype. This project tackles that challenge head-on by building and comparing multiple deep learning architectures for next-day price prediction.

Our best-performing model is an **Improved Transformer** that incorporates three novel modifications on top of a standard transformer encoder: **wavelet transform preprocessing** to strip high-frequency noise from raw price data, an **adaptive time window** that dynamically weights the importance of each day in the lookback window, and a **convolution-enhanced self-attention mechanism** that identifies the most relevant features across local temporal neighborhoods using learned convolutional kernels.

We tested this model on 10 years of Verizon (VZ) daily trading data (Nov 2015 -- Nov 2025) and compared it against LSTM baselines, reduced LSTMs, and classical ML methods. The improved transformer achieved a scaled RMSE of **0.2261** and generated a simulated annualized profit of over **17%** using a simple buy-when-predicted-up trading strategy.

We also conducted an extensive **sentiment analysis** feasibility study using FinBERT on news headlines sourced from the GDELT dataset, and explored macroeconomic features (oil prices, GDP, inflation, unemployment) from the FRED API. Neither significantly improved the transformer's predictive accuracy -- sentiment scores were near-neutral ~95% of the time, and macroeconomic signals were already encoded in price and volume.

---

## Features

- **Wavelet Transform Denoising** -- Discrete Wavelet Transform (DWT) with Donoho-Johnstone soft-thresholding to remove market noise while preserving meaningful price dynamics
- **Adaptive Time Window** -- A gating mechanism that assigns learned importance scores (0 to 1) to each day in the 60-day lookback, filtering out irrelevant historical data
- **Convolution-Enhanced Self-Attention** -- A 3-day convolutional kernel learns local temporal trends, then a dynamic weighting layer (S-Match) combines feature importance with attention scores
- **Hybrid Loss Function** -- Combines MSE prediction loss with L1 attention regularization to penalize unfocused attention patterns
- **Simulated Trading Bot** -- Automated backtesting that buys when the model predicts an upward move and tracks cumulative P&L
- **Sentiment Analysis Pipeline** -- FinBERT-based sentiment scoring on GDELT news headlines with GPU-optimized batch processing
- **Comprehensive Feature Engineering** -- SMA/EMA (20 & 50 day), relative volume, PE ratio, TTM EPS, and 20-day average price swing

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.12 |
| Deep Learning | PyTorch 2.x, TensorFlow/Keras |
| Data Collection | yfinance API, FRED API, GDELT (Google BigQuery) |
| NLP / Sentiment | FinBERT (Hugging Face Transformers) |
| Data Processing | Pandas, NumPy, PyWavelets (pywt) |
| Visualization | Matplotlib, Seaborn |
| Scaling | scikit-learn (StandardScaler, MinMaxScaler) |
| Environment | Google Colab (GPU runtime) |

---

## Model Architecture

```
Input Features (5 cols: Price, Volume, Relative Vol, Avg Swing, PE Ratio)
       |
  [Wavelet Denoising] -- DWT db4, level 2, soft-threshold
       |
  [Standard Scaling]
       |
  [Adaptive Time Window] -- Temporal gating via MLP + Softmax
       |
  [Conv-Enhanced Self-Attention]
       |-- MultiheadAttention (1 head)
       |-- Conv1D kernel=3 -> Sigmoid weighting (W_t)
       |-- S_match = W_t * Attention Output
       |
  [MLP Head] -- Linear(300->64) -> ReLU -> Dropout(0.2) -> Linear(64->1)
       |
  Next-Day Price Prediction
```

---

## Results

### Improved Transformer (Verizon -- 252-day test set)

| Metric | Value |
|---|---|
| Scaled RMSE | 0.2261 |
| Scaled MSE | 1.0337 |
| Trade Win Rate | 55.8% (77 wins / 61 losses) |
| Total Simulated Profit | $12.32 |
| Avg Profit per Trade | $0.089 (0.24%) |
| Annualized Return | ~17.28% |
| Largest Win | $1.59 |
| Largest Loss | $1.68 |

### Ablation Study Summary

| Configuration | Impact |
|---|---|
| + Wavelet Transform | Definitive improvement in accuracy |
| + Adaptive Time Window | Mixed results, marginal impact |
| + Dynamic Feature Importance (CNN) | Least effective for accuracy, but affected profit distribution |
| + Sentiment Analysis (FinBERT) | No significant improvement -- excluded from final model |

---

## Getting Started

### Prerequisites

- Python 3.10+
- PyTorch 2.x
- Google Colab (recommended) or local GPU

### Installation

```bash
git clone https://github.com/Mouner9360/Stock-Price-Prediction.git
cd Stock-Price-Prediction
pip install numpy pandas matplotlib seaborn scikit-learn torch pywt yfinance transformers
```

### Running the Improved Transformer

1. Open `Improved_Transformer_mod2_YearTestSize.ipynb` in Google Colab or Jupyter
2. Ensure `VZ_final_DF (1).csv` is in your working directory
3. To use our pre-trained weights, load `tuned_transformer_weights_with55profit.pth`
4. Run all cells to generate predictions, metrics, and the simulated trading report

### Running the LSTM Models

1. Open `New_columns_and_first_LSTM (5).ipynb`
2. The notebook handles feature engineering, training, backtesting, and forecasting

### Running Sentiment Analysis

1. Open `Sentiment Data Cleaning.ipynb` to preprocess GDELT data
2. Open `FinBERT Sentiment Analysis.ipynb` to generate sentiment scores
3. Use `Reduced LSTM with Sentiment.ipynb` to test with sentiment features

---

## Project Structure

```
Stock-Price-Prediction/
|-- Data_Collection_Capstone492_stock (1).ipynb   # yfinance data collection
|-- New_columns_and_first_LSTM (5).ipynb          # Feature engineering + LSTM models
|-- Improved_Transformer_mod2_YearTestSize.ipynb  # Improved transformer (main model)
|-- FinBERT Sentiment Analysis.ipynb              # Sentiment scoring pipeline
|-- Sentiment Data Cleaning.ipynb                 # GDELT data cleaning
|-- Reduced LSTM with Sentiment.ipynb             # LSTM + sentiment testing
|-- tuned_transformer_weights_with55profit.pth    # Pre-trained transformer weights
|-- VZ_final_DF (1).csv                           # Final engineered feature set
|-- VZ_PRICE_MOVEMENT_Data (1).csv                # Raw price/volume data
|-- verizon long.csv                              # Historical earnings data
|-- denoised_VZ_stock_data.csv                    # Wavelet-denoised dataset
|-- VZ_trades (1).csv                             # Simulated trading results
|-- *Sentiment*.csv                               # Various sentiment datasets
```

---

## Contributing

Pull requests are welcome. For major changes, please open an issue first.

---

## License

[MIT](LICENSE)

---

## Authors

**Mouner Wissa** -- Macroeconomic data collection & preprocessing (FRED API), feature engineering, report writing (abstract, sentiment analysis, background, model architecture depth, future work with LLMs)
- Portfolio: [mouner9360.github.io/portfolio](https://mouner9360.github.io/portfolio)
- LinkedIn: [linkedin.com/in/mouner-wissa-8493a1282](https://www.linkedin.com/in/mouner-wissa-8493a1282/)
- GitHub: [@Mouner9360](https://github.com/Mouner9360)

**Christian Sodora** -- Improved transformer model, wavelet transform, CNN feature importance, adaptive window, ablation study

**Afsheen Khan** -- Literature review, unmodified transformer testing with sentiment analysis

**Nahreg Rastguelenian** -- Sentiment analysis data pipeline (GDELT + FinBERT), LSTM sentiment testing
