# Fake Stock Data for DeepTrader Testing

This directory contains simulated stock market data for testing the DeepTrader model with controlled market conditions.

## Overview

The dataset simulates a 10-year period (2015-2025) with two stocks and market data:
- **Stock A**: Bear market (gradual decline)
- **Stock B**: Bull market (gradual rise)
- **Market**: Average of A and B, overall bullish

## Time Periods

| Period     | Start Date | End Date   | Start Index | End Index | Business Days |
|------------|------------|------------|-------------|-----------|---------------|
| Training   | 2015-01-01 | 2019-12-31 | 0           | 1303      | 1304          |
| Validation | 2020-01-01 | 2022-12-31 | 1304        | 2086      | 783           |
| Test       | 2023-01-01 | 2025-03-31 | 2087        | 2672      | 586           |

**Total**: 2673 business days

## Data Files

### 1. `stocks_data.npy`
- **Shape**: `(2, 2673, 1)`
- **Description**: Stock price data (close only)
- **Dimensions**:
  - Axis 0: Stock index (0=Stock A, 1=Stock B)
  - Axis 1: Time (business days)
  - Axis 2: Feature (close price)

### 2. `market_data.npy`
- **Shape**: `(2673, 1)`
- **Description**: Market index data calculated as (Stock A + Stock B) / 2
- **Dimensions**:
  - Axis 0: Time (business days)
  - Axis 1: Feature (close price)

### 3. `ror.npy`
- **Shape**: `(2, 2673)`
- **Description**: Rate of Return calculated as: `(close_t - close_{t-1}) / close_{t-1}`
- **Note**: First day (index 0) is 0 for both stocks
- **Dimensions**:
  - Axis 0: Stock index
  - Axis 1: Time (business days)

### 4. `industry_classification.npy`
- **Shape**: `(2, 2)`
- **Description**: Correlation matrix between stocks based on first 1000 days of returns
- **Content**:
  ```
  [[ 1.0000   -0.0466]
   [-0.0466    1.0000]]
  ```

## Market Characteristics

### Overall Performance (2015-2025)

| Asset   | Start Price | End Price | Total Return |
|---------|-------------|-----------|--------------|
| Stock A | $100.04     | $87.36    | -12.67%      |
| Stock B | $65.01      | $103.59   | +59.33%      |
| Market  | $82.53      | $95.47    | +15.69%      |

### Period-wise Performance

#### Training Period (2015-2019)
- Stock A: $100.04 → $94.25 (-5.79%)
- Stock B: $65.01 → $82.53 (+26.94%)
- Market: $82.53 → $88.39 (+7.10%)

#### Validation Period (2020-2022)
- Stock A: $94.28 → $90.64 (-3.86%)
- Stock B: $82.56 → $94.42 (+14.37%)
- Market: $88.42 → $92.53 (+4.65%)

#### Test Period (2023-2025Q1)
- Stock A: $90.62 → $87.36 (-3.60%)
- Stock B: $94.35 → $103.59 (+9.79%)
- Market: $92.49 → $95.47 (+3.23%)

## Cross Point

Stocks A and B cross at the following indices:
- Index 1867 (2022-02-28) - Validation period
- Index 1874 (2022-03-09) - Validation period
- Index 1875 (2022-03-10) - Validation period

After crossing, Stock B maintains higher prices than Stock A throughout the test period.

## Visualizations

### `stocks_AB.png`
Shows both Stock A (red, bear market) and Stock B (blue, bull market) with period backgrounds:
- Light gray: Training period
- Medium gray: Validation period
- Dark gray: Test period

### `market_data.png`
Shows the market index (A+B)/2 in green, displaying overall bullish trend.

## Generation

To regenerate the data:
```bash
conda run -n DeepTrader-pip python generate_fake_data.py
```

## Parameters Used

- **Stock A**: Linear decline from $100 to $80 with noise (volatility=0.008)
- **Stock B**: Linear rise from $65 to $100 with noise (volatility=0.008)
- **Random Seed**: 42 (for reproducibility)
- **Correlation**: -0.0466 (weak negative correlation between stocks)

## Use Cases

This simplified dataset is ideal for:
1. Testing model behavior with clear market trends
2. Debugging portfolio allocation strategies
3. Validating long/short strategies
4. Quick prototyping without downloading real data
5. Understanding model response to bear/bull market conditions
