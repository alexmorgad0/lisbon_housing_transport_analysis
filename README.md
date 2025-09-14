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
4. Can I build a model to estimate housing prices across the region?


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

I combined multiple data sources to build a unified dataset of housing prices and accessibility indicators across the Lisbon metropolitan area.

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
3) Build GTFS route model 
4) Applying the model to 3 Towns
5) Use Google Distance Matrix API (184 towns)  
6) Merge data and train the price prediction model

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

## 2.2 Geocode every town (lat/lon)
The main Kaggle housing dataset only included town names, not geographic coordinates.  
To calculate realistic commute times and model routes through the public transportation system, I needed the latitude and longitude of every town.  

I used the  `geopy` library to convert each town name into coordinates, then cached the results to avoid repeated API calls.


<details>
<summary>📍 Show Geocoding Code</summary>

```python
# Install dependencies as I was running it on Google Collab 
!pip -q install geopy tqdm unidecode

import os, pandas as pd
from geopy.geocoders import Nominatim # geocoding library
from geopy.extra.rate_limiter import RateLimiter # ensures that I dont get banned from making too many requests
from unidecode import unidecode # remove accents from names
from tqdm import tqdm

# File to cache results
CACHE_PATH = "geocode_towns_cache.csv"

# Start from cache if it exists, otherwise create new
if os.path.exists(CACHE_PATH):
    cache = pd.read_csv(CACHE_PATH)
else:
    cache = pd.DataFrame({"Town": towns_df["Town"], "lat": None, "lon": None})

# Configuring the Geocoder 
geolocator = Nominatim(user_agent="lisbon-housing-transport/1.0 (student project)")
geocode = RateLimiter(
    geolocator.geocode,
    min_delay_seconds=1.2,   # avoid getting blocked
    max_retries=2,
    swallow_exceptions=True
)
# Defining the geocoding logic and tries 3 different approaches
def geocode_town(town: str):
    # 1) Plain string with country filter
    loc = geocode(f"{town}, Portugal", country_codes="pt", language="pt")
    if loc:
        return loc.latitude, loc.longitude

    # 2) Structured query
    loc = geocode({"city": town, "country": "Portugal"}, language="pt")
    if loc:
        return loc.latitude, loc.longitude

    # 3) Try without accents
    t2 = unidecode(town)
    if t2 != town:
        loc = geocode(f"{t2}, Portugal", country_codes="pt", language="pt")
        if loc:
            return loc.latitude, loc.longitude

    return None, None

# Only geocode missing ones
mask = cache["lat"].isna() | cache["lon"].isna()
to_do = cache.loc[mask, "Town"].tolist()

# Run with progress bar
for town in tqdm(to_do, desc="Geocoding towns"):
    lat, lon = geocode_town(town)
    cache.loc[cache["Town"] == town, ["lat", "lon"]] = [lat, lon]

# Save/refresh cache
cache.to_csv(CACHE_PATH, index=False)
print("Cache saved:", CACHE_PATH)
print("Geocoded:", cache['lat'].notna().sum(), "/", len(cache))
```
</details> 

## 2.3 GTFS Route Model (earliest arrival to Marquês de Pombal)

I built an earliest-arrival router over merged GTFS Transportation feeds (CP, Fertagus, Metro, Carris, Margem Sul Metro, Transtejo, Transportes do Barreiro). All of this files had a table with all the Stops, and Stop Times.  
It finds the fastest itinerary from a property’s coordinates to Marquês de Pombal (08:00AM weekday), allowing rides and walking transfers.

<details>
<summary> Show GTFS routing code</summary>

```python
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# -------------------------------------------------------------------
# PRECONDITIONS
# - stops_merged have: ['stop_id','stop_name','stop_lat','stop_lon','feed']
# - stop_times_merged have: ['trip_id','arrival_time','departure_time','stop_id','stop_sequence','feed']
# If a namespaced stop id ("sid") doesn't exist yet, build it once:
if 'sid' not in stops_merged.columns:
    stops_merged = stops_merged.copy()
    stops_merged['sid'] = stops_merged['feed'].astype(str) + ':' + stops_merged['stop_id'].astype(str)

if 'sid' not in stop_times_merged.columns:
    stop_times_merged = stop_times_merged.copy()
    stop_times_merged['sid'] = stop_times_merged['feed'].astype(str) + ':' + stop_times_merged['stop_id'].astype(str)

# -------------------------------------------------------------------
# CONSTANTS & HELPERS
R = 6371000.0          # Earth radius (m)
WALK_M_PER_MIN = 75.0  # walking speed (m/min) ~ 4.5 km/h

def gtfs_time_to_s(t: str) -> int: # transforms time in seconds
    h, m, s = map(int, str(t).split(":"))
    return h*3600 + m*60 + s

def sec_to_hhmm(s):
    s = int(s)
    h = s // 3600
    m = (s % 3600) // 60
    return f"{h:02d}:{m:02d}"

FEED2MODE = {
    "cp": "train",
    "fertagus": "train",
    "metro": "metro",
    "carris": "bus",
    "mst": "tram",
    "transtejo": "ferry",
    "transportes_barreiro": "bus",
}

# -------------------------------------------------------------------
# Slimmed the stops table with only the needed columns
stops_min = (
    stops_merged
    .rename(columns={"stop_name":"name","stop_lat":"lat","stop_lon":"lon"})
    .loc[:, ["sid","stop_id","name","feed","lat","lon"]]
    .copy()
)
stops_min["lat"] = pd.to_numeric(stops_min["lat"], errors="coerce")
stops_min["lon"] = pd.to_numeric(stops_min["lon"], errors="coerce")
stops_min = stops_min.dropna(subset=["lat","lon"]).reset_index(drop=True)

# Keep a geographic window (Lisbon/Setúbal)
LIM = dict(lat_min=38.3, lat_max=39.1, lon_min=-9.6, lon_max=-8.4)
stops_min = stops_min[
    stops_min["lat"].between(LIM["lat_min"], LIM["lat_max"])
    & stops_min["lon"].between(LIM["lon_min"], LIM["lon_max"])
].reset_index(drop=True)

# BallTree for all stops (checks for all que qucikest and nearest stops)
all_coords = np.deg2rad(stops_min[["lat","lon"]].to_numpy())
btree_all  = BallTree(all_coords, metric="haversine")

# Separate BallTrees per operator, used to seed origins per feed
feed_trees, feed_index = {}, {}
for f in stops_min["feed"].dropna().unique():
    df = stops_min[stops_min["feed"] == f].reset_index(drop=True)
    if df.empty:
        continue
    arr = np.deg2rad(df[["lat","lon"]].to_numpy())
    feed_trees[f] = BallTree(arr, metric="haversine")
    feed_index[f] = df

def nearest_stops_from_point(lat, lon, k=6, max_dist_m=1500): # finds 6 closest stops overall, and sets a maximum distance walking of 1500 meters
    d_rad, idx = btree_all.query(np.deg2rad([[lat, lon]]), k=min(k, len(stops_min)))
    d_m = d_rad[0] * R
    keep = d_m <= max_dist_m
    if not keep.any():
        best = np.argmin(d_m)
        keep = np.zeros_like(d_m, dtype=bool)
        keep[best] = True
    idx = idx[0][keep]; d_m = d_m[keep]
    out = stops_min.iloc[idx][["sid","stop_id","name","feed","lat","lon"]].copy()
    out["walk_sec"] = (d_m / WALK_M_PER_MIN) * 60.0
    return out.reset_index(drop=True) # returns stops + walking time from the origin point

# for each feed finds up to 5 closest stops within a 1500 meters radius and keeps only short walking candidate per feed to make sure the model avoids biasing to one operator and ensure it considers multiple transport options.
def seeds_per_feed(lat, lon,
                   per_feed_k=5,
                   per_feed_radius_m=1500,
                   take_n_per_feed=2,
                   max_origin_walk_m=None):
    out = []
    q = np.deg2rad([[lat, lon]])
    for f, tree in feed_trees.items():
        df = feed_index[f]
        if df.empty:
            continue
        k = min(per_feed_k, len(df))
        d_rad, idx = tree.query(q, k=k)
        d_m = d_rad[0] * R
        max_r = per_feed_radius_m if max_origin_walk_m is None else min(per_feed_radius_m, max_origin_walk_m)
        keep = d_m <= max_r
        if not keep.any():
            continue
        idx = idx[0][keep]; d_m = d_m[keep]
        tmp = df.iloc[idx][["sid","stop_id","name","feed","lat","lon"]].copy()
        tmp["walk_sec"] = (d_m / WALK_M_PER_MIN) * 60.0
        tmp = tmp.nsmallest(take_n_per_feed, "walk_sec")
        out.append(tmp)
    if not out:
        return pd.DataFrame(columns=["sid","stop_id","name","feed","lat","lon","walk_sec"])
    return pd.concat(out, ignore_index=True)

# -------------------------------------------------------------------
# FOOTPATHS For every stops finds nearby stops within 600 meters and stores a list of neighbour stops and walking seconds and is used during routing for inter-stop walking transfers
def build_footpaths(radius_m=600):
    if len(stops_min) == 0:
        return {}
    rad = radius_m / R  # radius in radians for haversine
    inds, dists = btree_all.query_radius(all_coords, r=rad, return_distance=True, sort_results=False)
    ids = stops_min["sid"].to_numpy()
    foot = {}
    for i in range(len(ids)):
        src = ids[i]
        nbr_idx = inds[i]
        nbr_dist_m = dists[i] * R
        walk_sec = (nbr_dist_m / WALK_M_PER_MIN) * 60.0
        rows = []
        for j, dst_idx in enumerate(nbr_idx):
            if int(dst_idx) == i:
                continue
            rows.append((ids[int(dst_idx)], float(walk_sec[j])))
        foot[src] = rows
    return foot

footpaths = build_footpaths(radius_m=600)

# -------------------------------------------------------------------
# CONNECTIONS Builds connections between stops to change between different Transportation Operators
def build_connections(t_start="05:00:00", t_end="12:00:00"):
    start_s, end_s = gtfs_time_to_s(t_start), gtfs_time_to_s(t_end)

    st = stop_times_merged.merge(
        stops_min[["sid","feed"]],
        on=["sid","feed"], how="inner"
    ).copy()

    st["dep_s"] = st["departure_time"].apply(gtfs_time_to_s)
    st["arr_s"] = st["arrival_time"].apply(gtfs_time_to_s)
    st = st.dropna(subset=["dep_s","arr_s"])

    st = st[(st["dep_s"] <= end_s) & (st["arr_s"] >= start_s)]
    st = st.sort_values(["feed","trip_id","stop_sequence"])
    nxt = st.groupby(["feed","trip_id"]).shift(-1)

    dep_sid = st["sid"]
    arr_sid = nxt["sid"]

    trip_uid = st["feed"].astype(str) + ":" + st["trip_id"].astype(str)

    conn = pd.DataFrame({
        "trip_id":  st["trip_id"].values,
        "trip_uid": trip_uid.values,
        "feed":     st["feed"].values,
        "dep_sid":  dep_sid.values,
        "dep_time": st["departure_time"].values,
        "dep_s":    st["dep_s"].astype(int).values,
        "arr_sid":  arr_sid.values,
        "arr_time": nxt["arrival_time"].values,
        "arr_s":    nxt["arr_s"].values,
    }).dropna(subset=["arr_sid","arr_time","arr_s"]).copy()

    conn["arr_s"] = conn["arr_s"].astype(int)
    conn = conn[
        (conn["dep_s"] >= start_s) & (conn["dep_s"] <= end_s) & (conn["arr_s"] >= conn["dep_s"])
    ].sort_values("dep_s").reset_index(drop=True)

    return conn

conn = build_connections("05:00:00", "12:00:00")

# -------------------------------------------------------------------
# EARLIEST-ARRIVAL SEARCH FROM COORDINATE → COORDINATE. This finds all stops within max_dest_walk_m of the destination coordinate. Records final walk time for each. Uses seeds_per_feed() to pick initial stops you can walk # to from the origin; applies a walk penalty factor in cost to prefer riding sooner. Scans for rides, and pick the best destination stop. I choose to give a walk penalty to avoid the model to go for long walks when there # is rides sooner
def earliest_arrival_to_coord(
    src_lat, src_lon,
    dest_lat, dest_lon,
    depart_after="08:00:00",
    max_dest_walk_m=500,
    max_origin_walk_m=1200,
    ride_slack_sec=30,
    walk_slack_sec=30,
    per_feed_k=3,
    per_feed_radius_m=1200,
    take_n_per_feed=2,
    walk_penalty_factor=1.25
):
    # Destination stop candidates
    d_rad, idx = btree_all.query(np.deg2rad([[dest_lat, dest_lon]]), k=len(stops_min))
    d_m = d_rad[0] * R
    mask = d_m <= max_dest_walk_m
    if not mask.any():
        return {"note": "No stops within destination radius"}
    dest_sids = stops_min.iloc[idx[0][mask]]["sid"].tolist()
    dest_final_walk = {sid: float((m / WALK_M_PER_MIN) * 60.0) for sid, m in zip(dest_sids, d_m[mask])}

    # Seeds near origin
    depart_s = gtfs_time_to_s(depart_after)
    seeds = seeds_per_feed(
        src_lat, src_lon,
        per_feed_k=per_feed_k,
        per_feed_radius_m=per_feed_radius_m,
        take_n_per_feed=take_n_per_feed,
        max_origin_walk_m=max_origin_walk_m
    )
    if seeds.empty:
        return {"note": f"No stops within {max_origin_walk_m} m of origin"}

    INF = 10**12
    arr = pd.Series(INF, index=stops_min["sid"].unique(), dtype="int64")
    parent = {}

    # Initial walk to seeds (use penalized time for routing cost)
    for _, r in seeds.iterrows():
        sid = r["sid"]
        wsec_real = int(r["walk_sec"])
        wsec_eff  = int(r["walk_sec"] * walk_penalty_factor)
        t0 = depart_s + wsec_eff
        if t0 < arr.get(sid, INF):
            arr[sid] = t0
            parent[sid] = ("walk_origin", None, wsec_real)

    # Scan ride connections in time order
    for _, c in conn.iterrows():
        u, v = c["dep_sid"], c["arr_sid"]
        dep, arrv = int(c["dep_s"]), int(c["arr_s"])
        if arr.get(u, INF) + ride_slack_sec <= dep and arrv < arr.get(v, INF):
            arr[v] = arrv
            parent[v] = ("ride", u, c["trip_id"], c["feed"], c["dep_time"], c["arr_time"])
            # walking transfers from v
            for (w, wsec) in footpaths.get(v, []):
                wsec_eff = int(float(wsec) * walk_penalty_factor)
                cand = arrv + wsec_eff + walk_slack_sec
                if cand < arr.get(w, INF):
                    arr[w] = cand
                    parent[w] = ("walk", v, int(wsec))

    # Choose best destination
    best_sid, best_total = None, INF
    for sid in dest_sids:
        base = arr.get(sid, INF)
        if base >= INF:
            continue
        total = base + int(dest_final_walk[sid])
        if total < best_total:
            best_total, best_sid = total, sid
    if best_sid is None:
        return {"note": "No path found in time window"}

    # Backtrack sids
    path = []
    cur = best_sid
    while cur is not None and cur in parent:
        path.append(cur)
        step = parent[cur]
        cur = step[1] if isinstance(step, tuple) and len(step) > 1 else None
    path.reverse()

    sid2name = stops_min.set_index("sid")["name"].to_dict()
    sid2feed = stops_min.set_index("sid")["feed"].to_dict()

    # Build legs (walk/ride)
    legs = []
    t_cursor = depart_s

    # initial walk (use real walk seconds from parent)
    if path:
        first = path[0]
        if parent[first][0] == "walk_origin":
            w_sec_real = float(parent[first][2])
            legs.append({
                "type": "walk",
                "from": "Origin",
                "to": sid2name.get(first, first),
                "from_sid": None, "to_sid": first,
                "start": sec_to_hhmm(t_cursor),
                "end":   sec_to_hhmm(t_cursor + w_sec_real),
                "duration_min": round(w_sec_real/60.0, 1),
                "meters": round(w_sec_real/60.0 * WALK_M_PER_MIN, 0),
            })
            t_cursor += w_sec_real

    # intermediate legs
    for i in range(1, len(path)):
        prev_sid, sid = path[i-1], path[i]
        step = parent[sid]

        if step[0] == "ride":
            u, trip_id, feed, dep_t, arr_t = step[1], step[2], step[3], step[4], step[5]
            dep_s = gtfs_time_to_s(dep_t); arr_s = gtfs_time_to_s(arr_t)
            legs.append({
                "type": "ride",
                "mode": FEED2MODE.get(feed, feed),
                "feed": feed,
                "trip_id": trip_id,
                "from": sid2name.get(prev_sid, prev_sid),
                "to":   sid2name.get(sid, sid),
                "from_sid": prev_sid, "to_sid": sid,
                "dep_time": dep_t, "arr_time": arr_t,
                "duration_min": round((arr_s - dep_s)/60.0, 1),
            })
            t_cursor = arr_s

        elif step[0] == "walk":
            # real (unpenalized) walk seconds are stored at step[2]
            w_sec = float(step[2])
            legs.append({
                "type": "walk",
                "from": sid2name.get(prev_sid, prev_sid),
                "to":   sid2name.get(sid, sid),
                "from_sid": prev_sid, "to_sid": sid,
                "start": sec_to_hhmm(t_cursor),
                "end":   sec_to_hhmm(t_cursor + w_sec),
                "duration_min": round(w_sec/60.0, 1),
                "meters": round(w_sec/60.0 * WALK_M_PER_MIN, 0),
            })
            t_cursor += w_sec

    # final walk to destination coordinate (real time)
    final_w_sec = float((BallTree.haversine_distances(
        np.deg2rad([[stops_min.loc[stops_min['sid']==best_sid, 'lat'].values[0],
                     stops_min.loc[stops_min['sid']==best_sid, 'lon'].values[0]]]),
        np.deg2rad([[dest_lat, dest_lon]])
    )[0,0] * R) / WALK_M_PER_MIN * 60.0)

    legs.append({
        "type": "walk",
        "from": sid2name.get(best_sid, best_sid),
        "to": "Destination (coords)",
        "from_sid": best_sid, "to_sid": None,
        "start": sec_to_hhmm(t_cursor),
        "end":   sec_to_hhmm(t_cursor + final_w_sec),
        "duration_min": round(final_w_sec/60.0, 1),
        "meters": round(final_w_sec/60.0 * WALK_M_PER_MIN, 0),
    })
    t_cursor += final_w_sec

    return {
        "arrive_time": sec_to_hhmm(best_total),
        "total_min": round((best_total - depart_s)/60.0, 1),
        "dest_sid": best_sid,
        "dest_stop_name": sid2name.get(best_sid, best_sid),
        "final_walk_min": round(final_w_sec/60.0, 1),
        "path_sids": path,
        "path_stop_names": [sid2name.get(s, s) for s in path],
        "legs": legs,
    }
# When I call the function it prints the itinerary 
def print_itinerary(result):
    if "legs" not in result:
        print(result)
        return
    print(f"Arrive {result['arrive_time']}  (total {result['total_min']} min)")
    for i, leg in enumerate(result["legs"], 1):
        if leg["type"] == "ride":
            mode = leg.get("mode", leg.get("feed", "ride"))
            print(f"{i}. {mode.upper()}: {leg['from']} → {leg['to']}   "
                  f"{leg['dep_time']}–{leg['arr_time']}  ({leg['duration_min']} min)")
        else:
            print(f"{i}. WALK: {leg['from']} → {leg['to']}   "
                  f"{leg['start']}–{leg['end']}  (~{leg['duration_min']} min, ~{int(leg['meters'])} m)")

# -------------------------------------------------------------------
# EXAMPLE CALL
# Listing coordinates 
src_lat, src_lon = 38.7260, -9.1276   # Oeiras example

# Destination coordinates (Marquês de Pombal)
dest_lat, dest_lon = 38.7253, -9.1500

res = earliest_arrival_to_coord(
    src_lat, src_lon,
    dest_lat, dest_lon,
    depart_after="08:00:00",
    max_dest_walk_m=500,
    max_origin_walk_m=1200,
    ride_slack_sec=30,
    walk_slack_sec=30,
    per_feed_k=3,
    per_feed_radius_m=1200,
    take_n_per_feed=2,
    walk_penalty_factor=1.25
)
print_itinerary(res)

```
</details> 

## 2.4 Apply the Model to 3 Towns

While preparing to run the GTFS router for all towns, I realized that some areas were missing full coverage from the Carris GTFS feed, which I was unable to locate online.  
Because of this limitation, I decided to apply the model only to three towns where I was certain all the required GTFS data was available.  
This allowed me to validate that the routing logic worked correctly before moving on to the other towns using the Google Distance Matrix API.
  
<details>
<summary> Show Application of the model code</summary>

```python
import pandas as pd

CENTER_LAT, CENTER_LON = 38.7253, -9.1500  # Marquês de Pombal (approx)
TOWNS_3 = ["Penha de França", "Massamá e Monte Abraão", "Barreiro e Lavradio"]  # 3 towns I choose that I knew had all the data needed

# One representative coordinate per town, just in case there is idealista listings in here with different coordinates
town_coords = (
    df[df["Town"].isin(TOWNS_3)]
      .groupby("Town", as_index=False)[["lat","lon"]]
      .median()
)

# Calls the model that was built before
def run_router_once(lat, lon):
    return earliest_arrival_to_coord(
        lat, lon, CENTER_LAT, CENTER_LON,
        depart_after="08:00:00",
        max_dest_walk_m=500,
        max_origin_walk_m=1200,
        ride_slack_sec=30, walk_slack_sec=30,
        per_feed_k=3, per_feed_radius_m=1200, take_n_per_feed=2,
        walk_penalty_factor=1.25
    )

# formats the route into readable text
def legs_to_text(legs):
    parts = []
    for L in legs:
        if L["type"] == "ride":
            mode = L.get("mode", L.get("feed","RIDE")).upper()
            parts.append(f'{mode}: {L["from"]}→{L["to"]} {L["dep_time"]}-{L["arr_time"]}')
        else:
            parts.append(f'WALK: {L["from"]}→{L["to"]} {L["start"]}-{L["end"]} (~{int(L["meters"])} m)')
    return " | ".join(parts)

# sid -> (name, lat, lon) map for plotting itinerary points
_sid2 = stops_min.set_index("sid")[["name","lat","lon"]].to_dict(orient="index")

def legs_to_points(town, lat0, lon0, res):
    pts, seq = [], 0
    pts.append((town, seq, float(lat0), float(lon0), "origin")); seq += 1
    for leg in res.get("legs", []):
        to_sid = leg.get("to_sid")
        if to_sid and to_sid in _sid2:
            pts.append((town, seq, _sid2[to_sid]["lat"], _sid2[to_sid]["lon"], leg["type"]))
            seq += 1
    pts.append((town, seq, CENTER_LAT, CENTER_LON, "destination"))
    return pts

# Run and collect both summary rows and route points
rows, route_pts = [], []
for _, r in town_coords.iterrows():
    res = run_router_once(r.lat, r.lon)
    if "legs" in res:
        rows.append({
            "Town": r.Town,
            "travel_min": res["total_min"],
            "arrive_time": res["arrive_time"],
            "route_text": legs_to_text(res["legs"])
        })
        route_pts += legs_to_points(r.Town, r.lat, r.lon, res)
    else:
        rows.append({
            "Town": r.Town,
            "travel_min": None,
            "arrive_time": None,
            "route_text": res.get("note","no route")
        })

# 1) Saves only 1 row per town
town_routes = pd.DataFrame(rows).sort_values("travel_min", na_position="last").reset_index(drop=True)
print(town_routes)

# 2) Merge back with the main dataset with all the towns
df_out = df.merge(town_routes[["Town","travel_min","arrive_time"]], on="Town", how="left")


```
</details> 

## 2.5 Travel Times with the Google Maps API

Because some towns were missing coverage in my GTFS data feed, which I couldn’t locate online the remaining datasets I decided to use the Google Maps Directions API and Distance Matrix API to estimate travel times from the remaining 184 towns.

This allowed me to complete the dataset by:
- Getting transit routes (duration, arrival time, route steps, map points)
- Getting driving times and distances for comparison between the different Towns.

<details>
<summary>Show code</summary>

```python
# --- CONFIG ---
GOOGLE_API_KEY = "YOUR_API_KEY"          # For privacy reasons I occulted my API key
CENTER = (38.7253, -9.1500)               # Marquês de Pombal
towns_demo = ["Penha de França", "Massamá e Monte Abraão", "Barreiro e Lavradio"]

# Ensure data has the required columns
assert {"Town","lat","lon"} <= set(_base.columns)

# Get unique remaining towns (exclude the 3 demo towns)
other_towns = (
    _base.loc[~_base["Town"].isin(towns_demo), ["Town","lat","lon"]]
         .dropna()
         .groupby("Town", as_index=False)[["lat","lon"]]
         .median()
)

# --- Google Maps client ---
!pip -q install googlemaps
import googlemaps, pandas as pd, time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

gmaps = googlemaps.Client(key=GOOGLE_API_KEY)

# Makes sure we get our data at 8AM on the next weekday
def next_weekday_at(hour=8, minute=0, tz="Europe/Lisbon"):
    now = datetime.now(ZoneInfo(tz))
    target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    while target <= now or target.weekday() >= 5:  # skip weekends
        target = (target + timedelta(days=1)).replace(hour=hour, minute=minute)
    return target

DEPART_AT = next_weekday_at(8, 0, "Europe/Lisbon")

# --- Format helpers ---
def sec_to_hhmm(s:int) -> str:
    s = int(s); h = s//3600; m = (s%3600)//60
    return f"{h:02d}:{m:02d}"

def _fmt_epoch_hhmm(x):
    try: return datetime.fromtimestamp(int(x)).strftime("%H:%M")
    except: return None

# --- Format route steps into readable text
def build_route_text(steps):
    parts = []
    for st in steps:
        mode = (st.get("travel_mode") or "").upper()
        if mode == "TRANSIT":
            td = st.get("transit_details") or {}
            dep = (td.get("departure_stop") or {}).get("name","?")
            arr = (td.get("arrival_stop") or {}).get("name","?")
            line = (td.get("line") or {}).get("short_name") or (td.get("line") or {}).get("name") or "Transit"
            dep_t = _fmt_epoch_hhmm((td.get("departure_time") or {}).get("value"))
            arr_t = _fmt_epoch_hhmm((td.get("arrival_time") or {}).get("value"))
            parts.append(f"TRANSIT({line}): {dep}→{arr} {dep_t}-{arr_t}")
        elif mode == "WALKING":
            dur_s = (st.get("duration") or {}).get("value", 0)
            dist_m = (st.get("distance") or {}).get("value", 0)
            parts.append(f"WALK: {sec_to_hhmm(dur_s)} (~{int(dist_m)} m)")
        else:
            dur_s = (st.get("duration") or {}).get("value", 0)
            dist_m = (st.get("distance") or {}).get("value", 0)
            parts.append(f"{mode or 'STEP'}: {sec_to_hhmm(dur_s)} (~{int(dist_m)} m)")
    return " | ".join(parts)

# Decodes the geometry polyline of each step into (lat, lon) points
def step_points_with_type(step):
    from googlemaps.convert import decode_polyline
    pts = []
    mode = (step.get("travel_mode") or "").upper()
    seg = "ride" if mode == "TRANSIT" else ("walk" if mode == "WALKING" else mode.lower() or "other")
    poly = (step.get("polyline") or {}).get("points")
    if poly:
        for p in decode_polyline(poly):
            pts.append((p["lat"], p["lng"], seg))
    else:
        end = step.get("end_location")
        if end:
            pts.append((end["lat"], end["lng"], seg))
    return pts

#  Call Google Directions API and extracts Travel duration in minutes, arrive time, route text and points (sequence of the coordinates)
def google_route_for_town(lat: float, lon: float):
    resp = gmaps.directions(
        origin=(float(lat), float(lon)),
        destination=CENTER,
        mode="transit",
        departure_time=DEPART_AT
    )
    if not resp:
        return None

    leg = resp[0]["legs"][0]
    travel_min = round(leg["duration"]["value"] / 60, 1)
    arrive_time = (leg.get("arrival_time") or {}).get("text")
    if not arrive_time:
        dep_epoch = (leg.get("departure_time") or {}).get("value")
        if dep_epoch:
            arrive_time = _fmt_epoch_hhmm(int(dep_epoch) + int(leg["duration"]["value"]))

    rtext = build_route_text(leg.get("steps", []))

    seq = 0
    pts = [(float(lat), float(lon), "origin", seq)]; seq += 1
    for st in leg.get("steps", []):
        for plat, plon, seg in step_points_with_type(st):
            pts.append((plat, plon, seg, seq)); seq += 1
    pts.append((CENTER[0], CENTER[1], "destination", seq))

    return {"travel_min": travel_min, "arrive_time": arrive_time, "route_text": rtext, "points": pts}

# --- Loops trough all the remaining 184 towns 
rows = []; all_points = []
for _, r in other_towns.iterrows():
    out = google_route_for_town(r["lat"], r["lon"])
    time.sleep(0.15)
    if not out:
        rows.append({"Town": r["Town"], "travel_min_google": None, "arrive_time_google": None, "route_text_google": "no route"})
        continue
    rows.append({
        "Town": r["Town"],
        "travel_min_google": out["travel_min"],
        "arrive_time_google": out["arrive_time"],
        "route_text_google": out["route_text"]
    })
    all_points += [(r["Town"], seq, lat, lon, seg) for (lat, lon, seg, seq) in out["points"]]
# builds two final data frames
google_summary = pd.DataFrame(rows)
google_points  = pd.DataFrame(all_points, columns=["Town","PathOrder","lat","lon","segment_type"])

#  Loops through towns to get driving minutes and distance in km to city center
drive_rows = []
for _, r in other_towns.iterrows():
    try:
        dm = gmaps.distance_matrix(
            origins=[(float(r["lat"]), float(r["lon"]))],
            destinations=[CENTER],
            mode="driving",
            departure_time=DEPART_AT
        )
        el = dm["rows"][0]["elements"][0]
        if el.get("status") == "OK":
            dist_m = el["distance"]["value"]
            dur_s = el.get("duration_in_traffic", el["duration"])["value"]
            drive_rows.append({
                "Town": r["Town"],
                "drive_min": round(dur_s / 60, 1),
                "drive_km": round(dist_m / 1000, 1)
            })
        else:
            drive_rows.append({"Town": r["Town"], "drive_min": None, "drive_km": None})
    except:
        drive_rows.append({"Town": r["Town"], "drive_min": None, "drive_km": None})
    time.sleep(0.1)
# saves the driving time and distance from each town to city center
driving_df = pd.DataFrame(drive_rows)



```
</details> 

## 3. Housing Price Prediction Model

To estimate property prices, I trained a supervised regression model using the housing and accessibility features collected.  
I first tested a simple Linear Regression, but its performance was very poor (R² ≈ 0.17, RMSE ≈ €2,328/m²).  

I then trained a Random Forest Regressor, which performed much better, explaining about 75% of the variance (R² ≈ 0.75) with an error of €1,300/m².  
This error is still large and only suitable for showing broad trends — not individual property valuations.

> Disclaimer: 
> This model should **not be used to predict real house prices**.  
> Even though the Random Forest explains about 75% of the variance, it still has a large error (~€1,300/m²).  
> This is likely due to the dataset being very noisy and incomplete, for example, the sample from the Idealista API was very small and most of the data came from the Kaggle Dataset.


---

### 🧪 Attempt 1 — Linear Regression 
R² ≈ 0.17 , RMSE ≈ €2,328/m²

<details>
<summary>📉 Show code</summary>

```python
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Features & target
features = [
    "District","Town","Type",
    "TotalArea","TotalRooms","NumberOfBathrooms","Parking","Elevator",
    "travel_min_final","drive_min_final","drive_km_final","no_transit_route"
]
target = "price_per_m2" # we are predicting the price per m2

X = df[features].copy()
y = df[target].copy()

# Splits the features into categorical and numerical columns and wraps these two pipelines into a ColumnTransformer so preprocessing happens automatically for the right columns
cat_cols = ["District","Town","Type"]
num_cols = [c for c in features if c not in cat_cols]

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore"))
    ]), cat_cols),
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median"))
    ]), num_cols)
])

# --- Train/test split Ramdomly splits the dataset into a sample of 20% for test size and 80% train size
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Trains the full pipeline on the training data
linreg_pipe = Pipeline([
    ("prep", preprocessor),
    ("reg", LinearRegression())
])

linreg_pipe.fit(X_train, y_train)

# Evaluates the model and computes Root Mean Squared Error , wich is the average error in m2 and R2 wich is how much the variance is explained
y_pred = linreg_pipe.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R²:   {r2:.3f}")


```
</details> 

## Attempt 2 - Random Forest Regressor
R² ≈ 0.75 , RMSE ≈ €1299/m²
<details>
<summary>🌲 Show code</summary>

```python
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# --- Features & target ---
features = [
    "Town","Type",
    "TotalArea","TotalRooms","NumberOfBathrooms","Parking","Elevator",
    "travel_min_final","drive_min_final","drive_km_final","no_transit_route"
]
target = "price_per_m2"

X = df[features].copy()
y = df[target].copy()

# --- Preprocessor ---
cat_cols = ["Town","Type"]
num_cols = [c for c in features if c not in cat_cols]

preprocessor = ColumnTransformer([
    ("cat", Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ]), cat_cols),
    ("num", Pipeline([
        ("imp", SimpleImputer(strategy="median"))
    ]), num_cols),
])

# --- Train/test split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Random Forest model ---
rf = RandomForestRegressor(
    n_estimators=600,
    min_samples_leaf=2,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
)

# --- Full pipeline ---
rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("reg", rf)
])

# --- Train ---
rf_pipe.fit(X_train, y_train)

# --- Evaluate ---
y_pred = rf_pipe.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RF RMSE: {rmse:.2f}")
print(f"RF R²:   {r2:.3f}")

```
</details> 

## 3.1 Streamlit App

I built a separate [Streamlit](https://housing-price-app-gskomybxcbtkgxrfk5wpza.streamlit.app/) app to make the prediction model accessible in a user-friendly interface.

Instead of running code notebooks, this app lets users:

- Select a town, property type, area, rooms, and other features
- Instantly get the predicted price per m² from the trained model
- View the model’s estimated price in a clean and interactive UI

This makes it easy to explore the model’s output without needing any coding experience.

🔗 **App Repository:** [link-to-streamlit-repo](https://github.com/alexmorgad0/housing-price-app)

