import streamlit as st
import pandas as pd
import sqlite3
import plotly.express as px
import plotly.figure_factory as ff
import numpy as np
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression
import datetime

load_dotenv()

@st.cache_data
def load_data():
    df = pd.read_csv("Superstore.csv", encoding='latin1')
    conn = sqlite3.connect("superstore.db")
    df.to_sql("superstore", conn, if_exists="replace", index=False)
    conn.close()
    return df

def query_db(query):
    conn = sqlite3.connect("superstore.db")
    result = pd.read_sql(query, conn)
    conn.close()
    return result

df = load_data()
df['Order Date'] = pd.to_datetime(df['Order Date'], dayfirst=True)

st.title("Superstore Sales Dashboard")
st.markdown("Interactive sales analytics powered by SQL & Python")

# Sidebar filters
st.sidebar.header("Filters")
region = st.sidebar.multiselect("Region", df['Region'].unique(), default=list(df['Region'].unique()))
category = st.sidebar.multiselect("Category", df['Category'].unique(), default=list(df['Category'].unique()))
years = st.sidebar.multiselect("Year", sorted(df['Order Date'].dt.year.unique()), default=list(df['Order Date'].dt.year.unique()))

filtered = df[
    (df['Region'].isin(region)) &
    (df['Category'].isin(category)) &
    (df['Order Date'].dt.year.isin(years))
]

# Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Sales", f"${filtered['Sales'].sum():,.0f}")
col2.metric("Total Profit", f"${filtered['Profit'].sum():,.0f}")
col3.metric("Avg Discount", f"{filtered['Discount'].mean()*100:.1f}%")
col4.metric("Profit Margin", f"{filtered['Profit'].sum()/filtered['Sales'].sum()*100:.1f}%")

st.markdown("---")

# Row 1: Category & Region
col4, col5 = st.columns(2)
with col4:
    fig1 = px.bar(
        filtered.groupby('Category')[['Sales','Profit']].sum().reset_index(),
        x='Category', y=['Sales','Profit'], barmode='group',
        title='Sales & Profit by Category'
    )
    st.plotly_chart(fig1, use_container_width=True)

with col5:
    fig2 = px.bar(
        filtered.groupby('Region')['Profit'].sum().reset_index(),
        x='Region', y='Profit', color='Profit',
        color_continuous_scale='RdYlGn',
        title='Profit by Region'
    )
    st.plotly_chart(fig2, use_container_width=True)

# Row 2: Discount vs Profit & Sub-Category
col6, col7 = st.columns(2)
with col6:
    fig3 = px.scatter(
        filtered, x='Discount', y='Profit', color='Category',
        title='Discount vs Profit', opacity=0.5
    )
    st.plotly_chart(fig3, use_container_width=True)

with col7:
    sub_profit = filtered.groupby('Sub-Category')['Profit'].sum().reset_index().sort_values('Profit')
    fig4 = px.bar(
        sub_profit, x='Profit', y='Sub-Category',
        color='Profit', color_continuous_scale='RdYlGn',
        title='Profit by Sub-Category', orientation='h'
    )
    st.plotly_chart(fig4, use_container_width=True)

# Segment Analysis
st.markdown("---")
st.subheader("Segment Analysis")
col8, col9 = st.columns(2)

with col8:
    seg_sales = filtered.groupby('Segment')[['Sales','Profit']].sum().reset_index()
    fig5 = px.bar(seg_sales, x='Segment', y=['Sales','Profit'], barmode='group',
                  title='Sales & Profit by Segment')
    st.plotly_chart(fig5, use_container_width=True)

with col9:
    fig6 = px.pie(filtered.groupby('Segment')['Sales'].sum().reset_index(),
                  values='Sales', names='Segment', title='Sales Share by Segment')
    st.plotly_chart(fig6, use_container_width=True)

# Discount Impact Analysis
st.markdown("---")
st.subheader("Discount Impact Analysis")
threshold = st.slider("Discount threshold (%)", 0, 80, 20)
above = filtered[filtered['Discount'] >= threshold/100]['Profit'].mean()
below = filtered[filtered['Discount'] < threshold/100]['Profit'].mean()

col10, col11 = st.columns(2)
col10.metric(f"Avg Profit (Discount ≥ {threshold}%)", f"${above:,.0f}")
col11.metric(f"Avg Profit (Discount < {threshold}%)", f"${below:,.0f}")

fig7 = px.box(filtered, x=pd.cut(filtered['Discount'],
              bins=[0, threshold/100, 1], labels=[f'< {threshold}%', f'≥ {threshold}%']),
              y='Profit', color='Category', title=f'Profit Distribution by Discount Threshold ({threshold}%)')
st.plotly_chart(fig7, use_container_width=True)

# Time Series + Forecast
st.markdown("---")
st.subheader("Sales Trend & Forecast")

monthly = filtered.groupby(filtered['Order Date'].dt.to_period('M'))['Sales'].sum().reset_index()
monthly['Order Date'] = monthly['Order Date'].astype(str)
monthly['index'] = range(len(monthly))

X = monthly['index'].values.reshape(-1, 1)
y = monthly['Sales'].values
model = LinearRegression().fit(X, y)

future_idx = np.array(range(len(monthly), len(monthly)+6)).reshape(-1, 1)
future_dates = pd.period_range(
    start=pd.Period(monthly['Order Date'].iloc[-1], 'M') + 1, periods=6, freq='M'
).astype(str)
forecast = model.predict(future_idx)

last_actual = monthly[['Order Date', 'Sales']].iloc[[-1]].copy()
last_actual['Type'] = 'Forecast'
forecast_df = pd.DataFrame({'Order Date': future_dates, 'Sales': forecast, 'Type': 'Forecast'})
forecast_df = pd.concat([last_actual, forecast_df], ignore_index=True)
monthly['Type'] = 'Actual'
combined = pd.concat([monthly[['Order Date','Sales','Type']], forecast_df])

fig8 = px.line(combined, x='Order Date', y='Sales', color='Type',
               color_discrete_map={'Actual': '#378ADD', 'Forecast': '#E24B4A'},
               title='Monthly Sales Trend with 6-Month Forecast')
fig8.update_xaxes(tickangle=45)
st.plotly_chart(fig8, use_container_width=True)

# Geographic Analysis
st.markdown("---")
st.subheader("Geographic Analysis")

state_profit = filtered.groupby('State')['Profit'].sum().reset_index()

us_state_abbrev = {
    'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR',
    'California': 'CA', 'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE',
    'Florida': 'FL', 'Georgia': 'GA', 'Hawaii': 'HI', 'Idaho': 'ID',
    'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
    'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD',
    'Massachusetts': 'MA', 'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS',
    'Missouri': 'MO', 'Montana': 'MT', 'Nebraska': 'NE', 'Nevada': 'NV',
    'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM', 'New York': 'NY',
    'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
    'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC',
    'South Dakota': 'SD', 'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT',
    'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA', 'West Virginia': 'WV',
    'Wisconsin': 'WI', 'Wyoming': 'WY', 'District of Columbia': 'DC'
}

state_profit['State Code'] = state_profit['State'].map(us_state_abbrev)
state_profit['Profit_Display'] = state_profit['Profit'].apply(
    lambda x: f"${x:,.0f}" if x >= 0 else f"-${abs(x):,.0f}"
)

fig9 = px.choropleth(
    state_profit,
    locations='State Code',
    locationmode='USA-states',
    color='Profit',
    scope='usa',
    color_continuous_scale='RdYlGn',
    color_continuous_midpoint=0,
    title='Profit by State (Green = Profit, Red = Loss)',
    custom_data=['State', 'State Code', 'Profit_Display']
)

fig9.update_traces(
    hovertemplate="<b>%{customdata[0]}</b><br>State Code: %{customdata[1]}<br>Profit: %{customdata[2]}<extra></extra>"
)

fig9.update_coloraxes(
    colorbar_tickformat='$,.0f',
    colorbar_title='Profit'
)

st.plotly_chart(fig9, use_container_width=True)

# Correlation Heatmap
st.markdown("---")
st.subheader("Correlation Heatmap")
corr = filtered[['Sales','Quantity','Discount','Profit']].corr().round(2)
fig10 = ff.create_annotated_heatmap(
    z=corr.values, x=list(corr.columns), y=list(corr.index),
    colorscale='RdBu', showscale=True
)
st.plotly_chart(fig10, use_container_width=True)

# SQL Insights
st.markdown("---")
st.subheader("SQL Insights")
col12, col13 = st.columns(2)

with col12:
    st.markdown("**Top 5 Most Profitable Sub-Categories**")
    top5 = query_db("""
        SELECT "Sub-Category",
               ROUND(SUM(Profit), 0) AS Total_Profit,
               ROUND(SUM(Profit)/SUM(Sales)*100, 1) AS Margin_Pct
        FROM superstore
        GROUP BY "Sub-Category"
        ORDER BY Total_Profit DESC
        LIMIT 5
    """)
    st.dataframe(top5, use_container_width=True, hide_index=True)

with col13:
    st.markdown("**Top 5 Loss-Making Sub-Categories**")
    bottom5 = query_db("""
        SELECT "Sub-Category",
               ROUND(SUM(Profit), 0) AS Total_Profit,
               ROUND(AVG(Discount)*100, 1) AS Avg_Discount_Pct
        FROM superstore
        GROUP BY "Sub-Category"
        ORDER BY Total_Profit ASC
        LIMIT 5
    """)
    st.dataframe(bottom5, use_container_width=True, hide_index=True)

st.markdown("**Top 10 Most Profitable Customers**")
top_customers = query_db("""
    SELECT "Customer Name", "Segment",
           ROUND(SUM(Sales), 0) AS Total_Sales,
           ROUND(SUM(Profit), 0) AS Total_Profit
    FROM superstore
    GROUP BY "Customer Name"
    ORDER BY Total_Profit DESC
    LIMIT 10
""")
st.dataframe(top_customers, use_container_width=True, hide_index=True)

# RFM Analysis
st.markdown("---")
st.subheader("RFM Customer Segmentation")

snapshot_date = filtered['Order Date'].max() + pd.Timedelta(days=1)
rfm = filtered.groupby('Customer Name').agg(
    Recency=('Order Date', lambda x: (snapshot_date - x.max()).days),
    Frequency=('Order ID', 'nunique'),
    Monetary=('Sales', 'sum')
).reset_index()

rfm['R_Score'] = pd.qcut(rfm['Recency'], 4, labels=[4,3,2,1]).astype(int)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 4, labels=[1,2,3,4]).astype(int)
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 4, labels=[1,2,3,4]).astype(int)
rfm['RFM_Score'] = rfm['R_Score'] + rfm['F_Score'] + rfm['M_Score']

def segment(score):
    if score >= 10:
        return 'Champions'
    elif score >= 8:
        return 'Loyal'
    elif score >= 6:
        return 'Potential'
    elif score >= 4:
        return 'At Risk'
    else:
        return 'Lost'

rfm['Segment'] = rfm['RFM_Score'].apply(segment)

col_rfm1, col_rfm2 = st.columns(2)
with col_rfm1:
    seg_count = rfm['Segment'].value_counts().reset_index()
    seg_count.columns = ['Segment', 'Count']
    fig_rfm1 = px.bar(seg_count, x='Segment', y='Count',
                      color='Segment', title='Customer Segments')
    st.plotly_chart(fig_rfm1, use_container_width=True)

with col_rfm2:
    fig_rfm2 = px.scatter(rfm, x='Recency', y='Monetary',
                          size='Frequency', color='Segment',
                          title='RFM Scatter — Recency vs Monetary',
                          hover_name='Customer Name')
    st.plotly_chart(fig_rfm2, use_container_width=True)

st.markdown("**RFM Segment Summary**")
rfm_summary = rfm.groupby('Segment').agg(
    Customers=('Customer Name', 'count'),
    Avg_Recency=('Recency', 'mean'),
    Avg_Frequency=('Frequency', 'mean'),
    Avg_Monetary=('Monetary', 'mean')
).round(1).reset_index()
st.dataframe(rfm_summary, use_container_width=True, hide_index=True)

# Anomaly Detection
st.markdown("---")
st.subheader("Anomaly Detection")

Q1 = filtered['Profit'].quantile(0.25)
Q3 = filtered['Profit'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

filtered_copy = filtered.copy()
filtered_copy['Anomaly'] = filtered_copy['Profit'].apply(
    lambda x: 'Anomaly' if x < lower or x > upper else 'Normal'
)

anomaly_count = filtered_copy['Anomaly'].value_counts()
col_a1, col_a2 = st.columns(2)
col_a1.metric("Normal Transactions", f"{anomaly_count.get('Normal', 0):,}")
col_a2.metric("Anomalies Detected", f"{anomaly_count.get('Anomaly', 0):,}")

fig_anomaly = px.scatter(
    filtered_copy, x='Sales', y='Profit',
    color='Anomaly',
    color_discrete_map={'Normal': '#378ADD', 'Anomaly': '#E24B4A'},
    title='Anomaly Detection — Profit Outliers (IQR Method)',
    hover_data=['Category', 'Sub-Category', 'State']
)
fig_anomaly.add_hline(y=upper, line_dash='dash', line_color='orange', annotation_text='Upper bound')
fig_anomaly.add_hline(y=lower, line_dash='dash', line_color='orange', annotation_text='Lower bound')
st.plotly_chart(fig_anomaly, use_container_width=True)

st.markdown("**Top 10 Anomalous Transactions**")
anomalies = filtered_copy[filtered_copy['Anomaly'] == 'Anomaly'][
    ['Order ID', 'Category', 'Sub-Category', 'State', 'Sales', 'Profit', 'Discount']
].sort_values('Profit').head(10).round(2)
st.dataframe(anomalies, use_container_width=True, hide_index=True)