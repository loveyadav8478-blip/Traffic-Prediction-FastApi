"""
Run this from your API folder to test FastAPI DIRECTLY
bypassing Spring Boot completely:
  python test_direct.py
"""
import requests

BASE_URL = "http://127.0.0.1:8000"
API_KEY  = "aa5d066ad3983b27133902f4931636477d7df15ac15188a756e7b9afa46fd35e"

# Login
r = requests.post(f"{BASE_URL}/login", json={"username":"admin","password":"loveyadav@1"})
token = r.json().get("access_token")
print(f"Login: {'✅' if token else '❌'}\n")

headers = {
    "Authorization": f"Bearer {token}",
    "x-api-key": API_KEY,
    "Content-Type": "application/json"
}

# Test exact Mumbai→Pune Friday 7PM scenario
tests = [
    {"label": "Mumbai→Pune Fri 7PM Clear",    "source":"Mumbai","destination":"Pune","hour":19,"day_of_week":4,"weather":"Clear",  "distance_km":148.0,"temp_cel":32.0},
    {"label": "Mumbai→Pune Fri 7PM Clear h=7","source":"Mumbai","destination":"Pune","hour":7, "day_of_week":4,"weather":"Clear",  "distance_km":148.0,"temp_cel":32.0},
    {"label": "Delhi→Ghaziabad Mon 9AM Rain",  "source":"Delhi", "destination":"Ghaziabad","hour":9,"day_of_week":0,"weather":"Rain","distance_km":28.0,"temp_cel":28.0},
    {"label": "Mumbai→Pune Fri 8PM Rain",      "source":"Mumbai","destination":"Pune","hour":20,"day_of_week":4,"weather":"Rain",  "distance_km":148.0,"temp_cel":28.0},
    {"label": "Mumbai→Pune Mon 9AM Clouds",    "source":"Mumbai","destination":"Pune","hour":9, "day_of_week":0,"weather":"Clouds","distance_km":148.0,"temp_cel":30.0},
]

for t in tests:
    label = t.pop("label")
    r = requests.post(f"{BASE_URL}/predict-route", json=t, headers=headers, timeout=10)
    if r.status_code == 200:
        best = r.json().get("best_route", {})
        tl   = best.get("traffic_label","?")
        conf = best.get("confidence", 0)
        fu   = best.get("features_used", {})
        icon = {"Low":"🟢","Medium":"🟡","High":"🔴"}.get(tl,"⚪")
        print(f"{icon} {label}")
        print(f"   → {tl} ({conf:.0%}) | hour={fu.get('hour')} dow={fu.get('day_of_week')} peak={fu.get('is_peak_hour')} weekend={fu.get('is_weekend')} weather={fu.get('weather')} city={fu.get('city')}")
    else:
        print(f"❌ {label}: {r.status_code} {r.text[:100]}")
    print()
