#  Lisbon Housing & Transportation Analysis

Welcome to my analysis of housing affordability and transportation accessibility across the **Lisbon metropolitan area**.

This project was created to explore where people can live more affordably while still maintaining good access to the city center. It combines housing prices, commute times, and transport network coverage into a single analysis.

The core data came from a [Kaggle](https://www.kaggle.com/datasets/luvathoms/portugal-real-estate-2024/data) dataset of property listings, enriched with live data collected via the [Idealista](https://www.idealista.pt) API. Transportation data came from multiple [GTFS](https://gtfs.org) feeds, complemented by travel times fetched from the [Google Maps](https://developers.google.com/maps) APIs.

> **Disclaimer:** This project was created for educational purposes only.


---

#  The Questions

This project set out to answer:

1. Which towns around Lisbon offer the lowest housing prices per m²?
2. How well connected are these towns to the city center by public transport?
3. Which areas offer the best balance of affordability and accessibility?
4. Can we build a model to estimate housing prices across the region?


---

# 🛠 Tools I Used

- **Python** ([Google Colab](https://colab.research.google.com))
  - `pandas`, `numpy` — data manipulation  
  - `matplotlib` — visualizations
  - `geopy` — get town coordinates
  - `googlemaps` — route requests
  - `scikit-learn` —  Random Forest model for housing price prediction  
  - `requests` — make API calls ([Google](https://developers.google.com/maps), [Idealista](https://www.idealista.pt))
- **Power BI** — Dashboards for price vs accessibility analysis
- **[Streamlit](https://streamlit.io)** — Interactive web app to explore my Prediction Model
- **[GitHub](https://github.com)**  

---

# 1. Data Collection & Manipulation

We combined multiple data sources to build a unified dataset of housing prices and accessibility indicators across the Lisbon metropolitan area.

- Collected 40k+ property listings from a [Kaggle](https://www.kaggle.com/datasets/luvathoms/portugal-real-estate-2024/data) dataset  
- Retrieved 200 additional live listings from the [Idealista](https://www.idealista.pt) API  
- Fetched geographic coordinates (latitude/longitude) for 187 towns using the [Google Maps Geocoding API](https://developers.google.com/maps/documentation/geocoding)  
- Gathered public transport schedule data from [GTFS](https://gtfs.org) feeds  
- Obtained travel times (car and public transport) for each town using the [Google Distance Matrix API](https://developers.google.com/maps/documentation/distance-matrix)

All datasets were standardized, cleaned, and merged to produce a single master table used for further analysis, modeling, and dashboard creation.

# 2. Building a Commute Model (step-by-step with code)

This section shows the full pipeline used to turn raw listings into a dataset with commute times and price predictions.  
Steps:
1) Collect listings from the Idealista API  
2) Geocode town names  
3) Build GTFS route model (3 towns)  
4) Use Google Distance Matrix API (184 towns)  
5) Merge data and train the price prediction model

---

## 2.1 Collect listings from the Idealista API
To complement the original Kaggle dataset, which contained around 40,000 listings, I wanted to gather a small sample of more recent and localized data directly from the Idealista API.  

The goal was to:
- Validate if the price patterns in the Kaggle dataset were still consistent with the current market  
- Experiment with integrating real-world APIs into a data pipeline as part of this educational project  

Due to API restrictions, only about 200 listings were collected.

> ⚠️ **Note:** Idealista’s API is partner-only. You need your own `CLIENT_ID` and `CLIENT_SECRET`.  
> I only captured ~200 listings due to access/rate limits.


<details>
<summary>Show Idealista API code</summary>

```python
# --- imports ---
import os, time, base64, getpass, requests
import pandas as pd

# --- config ---
COUNTRY = "pt"  
API_BASE = f"https://api.idealista.com/3.5/{COUNTRY}"
OAUTH_URL = "https://api.idealista.com/oauth/token"
SEARCH_URL = f"{API_BASE}/search"

# --- credentials (choose ONE approach) ---
# A) interactive (good for demos)
CLIENT_ID = input("Idealista API Key (client_id): ").strip()
CLIENT_SECRET = getpass.getpass("Idealista API Secret (client_secret): ").strip()

# B) or env vars (better for automation)
# CLIENT_ID = os.getenv("IDEALISTA_CLIENT_ID")
# CLIENT_SECRET = os.getenv("IDEALISTA_CLIENT_SECRET")

assert CLIENT_ID and CLIENT_SECRET, "Missing Idealista credentials."

# --- token (client-credentials) ---
def get_token(client_id: str, client_secret: str) -> str:
    data = {"grant_type": "client_credentials", "scope": "read"}
    r = requests.post(OAUTH_URL, data=data, auth=(client_id, client_secret), timeout=30)
    r.raise_for_status()
    return r.json()["access_token"]

access_token = get_token(CLIENT_ID, CLIENT_SECRET)
print("Token acquired (preview):", access_token[:12], "…")

# --- paginated search (multipart/form-data) ---
def search_idealista_sale(
    token: str,
    center_lat: float,
    center_lon: float,
    distance_m: int = 20000,
    property_type: str = "homes",
    max_items: int = 50,
    max_pages: int = 4,
    extra_params: dict | None = None,
    sleep_seconds: float = 1.2,
    language: str = "en"  
) -> pd.DataFrame:

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    rows = []

    for page in range(1, max_pages + 1):
        form = {
            "center": f"{center_lat},{center_lon}",
            "distance": distance_m,
            "operation": "sale",
            "propertyType": property_type,
            "maxItems": max_items,   # 1–50
            "numPage": page,         # starts at 1
            "language": language
        }
        if extra_params:
            form.update(extra_params)

        files = {k: (None, str(v)) for k, v in form.items()}
        r = requests.post(SEARCH_URL, headers=headers, files=files, timeout=60)
        if r.status_code >= 400:
            print("Request failed. Status:", r.status_code)
            print("Response text:", r.text[:800])
            r.raise_for_status()

        payload = r.json()
        elements = payload.get("elementList", [])
        rows.extend(elements)

        if not elements or len(elements) < int(max_items):
            break

        time.sleep(sleep_seconds)

    return pd.DataFrame(rows)

# --- Calling the API (Marques de Pombal approx) ---
LISBON_LAT, LISBON_LON = 38.7079, -9.1366

df_sale = search_idealista_sale(
    token=access_token,
    center_lat=LISBON_LAT,
    center_lon=LISBON_LON,
    distance_m=20000,
    property_type="homes",
    max_items=50,
    max_pages=4,
    extra_params={
        # "minPrice": 100000, "maxPrice": 600000,
        # "minSize": 40, "maxSize": 150,
        # "sort": "priceUp"  # priceUp | priceDown | publicationDate
    },
    language="en"
)

```
</details> 

## 2.1 Geocode every town (lat/lon)

```python
# Geocode 187 town names → lat/lon
import os, pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from unidecode import unidecode
from tqdm import tqdm

tqdm.pandas()

# df_towns must have a column: 'town_name'
# e.g., df_towns = pd.DataFrame({'town_name': ['Queluz', 'Setúbal', ...]})

geolocator = Nominatim(user_agent="lisbon-housing-transport")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1, max_retries=2, error_wait_seconds=2.0)

def geocode_town(name: str):
    q = f"{unidecode(name)}, Portugal"
    loc = geocode(q)
    if not loc:
        return pd.Series([None, None])
    return pd.Series([loc.latitude, loc.longitude])

df_towns[['lat', 'lon']] = df_towns['town_name'].progress_apply(geocode_town)
# Keep only rows with coordinates
df_towns = df_towns.dropna(subset=['lat','lon']).reset_index(drop=True)


