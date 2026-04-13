import json
import urllib.request
import urllib.parse
import re
from src.core import config  # Or 'import config' depending on your file structure

# ---------------------------------------------------------------------------
# Geocoding — Azure Maps (optional, skipped if AZURE_MAPS_KEY is unset)
# ---------------------------------------------------------------------------

def clean_address_for_geo(address: str) -> str:
    if not address:
        return ""
    
    # 1. Split into lines
    lines = address.split('\n')
    
    filtered_lines = []
    for line in lines:
        l = line.strip()
        
        # Skip lines that are likely NOT part of the physical address
        if any(x in l for x in ["Dir ", "Mgr ", "Main:", "Rx:", "Phone:", "Tel:", "Manager"]):
            continue
            
        # Skip lines that are just phone numbers (e.g., (714) 282-7064)
        if re.search(r'\(\d{3}\) \d{3}-\d{4}', l):
            continue
            
        filtered_lines.append(l)
    
    # 2. Join and remove any double spaces
    clean = " ".join(filtered_lines)
    
    # 3. Final safety: If the LLM left "Vons" or "Store" inside the address, 
    # and we are already passing it as store_name, remove it here to avoid duplication.
    clean = clean.replace("Store 2216", "").strip()
    
    return clean


def geocode(address: str, store_name: str = "") -> tuple[float | None, float | None]:
    """
    Look up lat/lon for a store address using Azure Maps Fuzzy Search API.
    """
    if not config.AZURE_MAPS_KEY or not address:
        return None, None
        
    # --- CLEANING STEP ---
    # We clean the address to remove "Dir Darlene Harlan" etc.
    cleaned_addr = clean_address_for_geo(address)
    
    # Combine with store name for better accuracy
    query = f"{store_name} {cleaned_addr}".strip()
    
    try:
        # Construct URL with encoded query
        encoded_query = urllib.parse.quote(query)
        url = (
            f"{config.AZURE_MAPS_URL}"
            f"?api-version=1.0"
            f"&subscription-key={config.AZURE_MAPS_KEY}"
            f"&query={encoded_query}"
            f"&limit=1"
            f"&countrySet=CA,US"
        )
        
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            
        # --- DEBUG START ---
        # This will show you if 'results' is empty or if Azure returned an error
        results = data.get("results", [])
        if not results:
            print(f"  [GEO]  No results found for: '{query}'")
            # print(f"  [GEO]  Full Response: {data}") # Uncomment if you want to see the whole JSON
        # --- DEBUG END ---
        if results:
            pos = results[0]["position"]
            return pos["lat"], pos["lon"]
            
    except Exception as e:
        print(f"  [GEO]  geocode failed for '{query}': {e}")
        
    return None, None

