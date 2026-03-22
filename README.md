# Superstore Sales Dashboard

An interactive sales analytics dashboard built with Python, SQL, and Streamlit — analyzing 9,994 retail transactions across the U.S.

## Features

- **KPI Metrics** — Total Sales, Profit, Avg Discount, Profit Margin
- **Category & Region Analysis** — Sales and profit breakdown by category and region
- **Segment Analysis** — Consumer, Corporate, Home Office comparison
- **Discount Impact Analysis** — Interactive slider to explore how discounts affect profitability
- **Sales Trend & Forecast** — Monthly sales trend with 6-month Linear Regression forecast
- **Geographic Analysis** — Interactive U.S. choropleth map showing profit by state
- **Correlation Heatmap** — Relationship between Sales, Quantity, Discount, and Profit
- **SQL Insights** — Top 5 most profitable and loss-making sub-categories via SQLite queries
- **RFM Customer Segmentation** — Customers segmented into Champions, Loyal, Potential, At Risk, Lost
- **Anomaly Detection** — IQR-based outlier detection on profit transactions

## Tech Stack

- **Python** — Pandas, NumPy, scikit-learn, Plotly, Streamlit
- **SQL** — SQLite for data querying and insights
- **Machine Learning** — Linear Regression (forecast), IQR Anomaly Detection, RFM Segmentation

## Key Insights

- Technology is the most profitable category with a 17.4% profit margin
- Central region has the lowest profit margin due to excessive discounting (avg 24%)
- Discounts above 20% result in negative average profit (-$9)
- Tables sub-category generates the most losses (-$17,725) with 26% avg discount
- Texas is the highest loss-making state (-$25,729)
- 193 customers classified as Champions with avg monetary value of $4,761

## Dataset

- Source: [Superstore Sales Dataset — Kaggle](https://www.kaggle.com/datasets/vivek468/superstore-dataset-final)
- 9,994 rows, 21 columns
- U.S. retail transactions from 2011 to 2014

## Installation
```bash
git clone https://github.com/altanozd/superstore-dashboard.git
cd superstore-dashboard
pip install -r requirements.txt
streamlit run app.py
```

## Requirements
```
streamlit
pandas
numpy
plotly
scikit-learn
python-dotenv
```