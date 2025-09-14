# 🏙️ Lisbon Housing & Transportation Analysis

Welcome to my analysis of housing affordability and transportation accessibility across the **Lisbon metropolitan area**.

This project was created to explore where people can live more affordably while still maintaining good access to the city center. It combines housing prices, commute times, and transport network coverage into a single analysis.

The core data came from a [Kaggle](https://www.kaggle.com/datasets/luvathoms/portugal-real-estate-2024/data) dataset of property listings, enriched with live data collected via the [Idealista](https://www.idealista.pt) API. Transportation data came from multiple [GTFS](https://gtfs.org) feeds, complemented by travel times fetched from the [Google Maps](https://developers.google.com/maps) APIs.

---

# ❓ The Questions

This project set out to answer:

1. Which towns around Lisbon offer the lowest housing prices per m²?
2. How well connected are these towns to the city center by public transport?
3. Can we build a model to estimate housing prices across the region?
4. Which areas offer the best balance of affordability and accessibility?

---

# 🛠 Tools I Used

- **Python** ([Google Colab](https://colab.research.google.com))
  - `pandas`, `numpy` — data wrangling  
  - `matplotlib`, `seaborn` — visualizations  
  - `scikit-learn` — machine learning price model  
  - `requests` — APIs ([Google](https://developers.google.com/maps), [Idealista](https://www.idealista.pt))
- **Power BI** — Dashboards for price vs accessibility analysis
- **[Streamlit](https://streamlit.io)** — Interactive web app to explore towns
- **[GitHub](https://github.com)** — Version control and project hosting

---

# 📦 1. Data Collection

We started with a [Kaggle](https://www.kaggle.com) dataset of 40k+ property listings in the Lisbon/Setúbal districts and combined it with a small (~200 listings) sample collected from the [Idealista](https://www.idealista.pt) API.

```python
import pandas as pd
import requests, json

# Load Kaggle dataset
df_kaggle = pd.read_csv('kaggle_listings.csv')

# Fetch Idealista listings
url = "https://api.idealista.com/3.5/es/search"
headers = {"Authorization": "Bearer YOUR_TOKEN"}
params = {"center": "38.7169,-9.139","propertyType":"homes","maxItems":200}
r = requests.get(url, headers=headers, params=params)
df_idealista = pd.json_normalize(r.json()['elementList'])

# Combine
df = pd.concat([df_kaggle, df_idealista], ignore_index=True)
