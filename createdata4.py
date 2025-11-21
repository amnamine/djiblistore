import json
import csv
import random
import re
import uuid

# ==========================================
# CONFIGURATION
# ==========================================
INPUT_FILE = 'scraping4.json'
OUTPUT_FILE = 'dataset_train4.csv'

TARGET_DATASET_SIZE = 3000 

# === TIERED BOOSTING (The "Secret Sauce") ===
# HIGH (15x): Rare items (Tablets/Modems) need to scream loud.
HIGH_BOOST = ["Tablette", "Routeur_Modem"]

# MEDIUM (3x): Phones are distinct enough, just need a small push.
MEDIUM_BOOST = ["Smartphone"] 

# LOW (0.2x): Accessories are "Noise". We aggressively reduce them
# so they don't drown out the important products.
# (Implicitly handled in the loop logic below)

CATEGORY_KEYWORDS = {
    "Smartphone": ["zte", "tecno", "oppo", "samsung", "realme", "blade", "nubia", "pova", "spark", "infinix", "galaxy", "redmi", "xiaomi", "v60", "a75", "a35"],
    "Routeur_Modem": ["d-link", "tcl", "modem", "box", "dwr", "wifi", "4g", "routeur"],
    "Tablette": ["tablette", "tab", "d-tech", "ipad"],
    "Accessoire_Audio": ["earbuds", "ecouteur", "airpods", "casque", "kit", "bluetooth", "hoco", "revaleo", "audio"],
    "Accessoire_Charge": ["cable", "chargeur", "power bank", "usb", "type-c", "lightning", "batterie"],
    "Accessoire_Auto": ["support", "car", "voiture", "fm", "transmitter"]
}

INTENTS_PREFIX = ["achat", "acheter", "prix", "combien coute", "chercher", "trouver", "le", "la", "les", "promo", "nouveau", "voir"]
INTENTS_SUFFIX = ["algerie", "djezzy", "pas cher", "en ligne", "livraison", "original", "2025", "promo", "solde", "magasin", "disponible"]

# ==========================================
# HELPER FUNCTIONS
# ==========================================

def clean_text(text):
    if not text: return ""
    text = re.sub(r'<[^>]+>', '', text) 
    text = re.sub(r'[^\w\s]', ' ', text) 
    return text.strip()

def clean_price(price_str):
    if not price_str: return "0 DA"
    clean = str(price_str).replace('&nbsp;', ' ').replace('&nbsp', ' ').replace('\xa0', ' ')
    clean = re.sub(r'\s+', ' ', clean).strip()
    return clean

def get_category(text):
    text_lower = text.lower()
    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(k in text_lower for k in keywords):
            return cat
    return "Accessoire_General"

def augment_query(base_query):
    method = random.choices(
        ["raw", "typo", "prefix", "suffix", "combination"],
        weights=[0.50, 0.20, 0.10, 0.10, 0.10], 
        k=1
    )[0]

    query = base_query.lower()
    
    if method == "raw":
        return query
    elif method == "typo":
        if len(query) < 5: return query 
        if random.random() > 0.5:
            idx = random.randint(0, len(query) - 2)
            return query[:idx] + query[idx+1] + query[idx] + query[idx+2:]
        else:
            idx = random.randint(0, len(query) - 1)
            return query[:idx] + query[idx+1:]
    elif method == "prefix":
        return f"{random.choice(INTENTS_PREFIX)} {query}"
    elif method == "suffix":
        return f"{query} {random.choice(INTENTS_SUFFIX)}"
    elif method == "combination":
        return f"{random.choice(INTENTS_PREFIX)} {query} {random.choice(INTENTS_SUFFIX)}"
    return query

# ==========================================
# MAIN GENERATOR
# ==========================================

def create_large_dataset():
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] {INPUT_FILE} not found.")
        return

    # --- STEP 1: DEDUPLICATE ---
    products_map = {}
    print(f"[1/3] Deduplicating {len(raw_data)} raw entries...")
    
    for item in raw_data:
        brand = clean_text(item.get("title", ""))       
        model = clean_text(item.get("description", "")) 
        
        if model.lower().startswith(brand.lower()):
            clean_name = model
        else:
            clean_name = f"{brand} {model}"
            
        clean_name = clean_name.strip()
        unique_key = clean_name.lower().replace(" ", "")
        
        if unique_key not in products_map and unique_key != "":
            products_map[unique_key] = {
                "id": str(uuid.uuid4())[:8],
                "brand": brand,
                "model": model,
                "name": clean_name,
                "category": get_category(clean_name),
                "price": clean_price(item.get("price", "0 DA"))
            }

    unique_products = list(products_map.values())
    print(f"      -> Found {len(unique_products)} UNIQUE products.")

    # --- STEP 2: GENERATE BALANCED DATASET ---
    dataset_rows = []
    base_rows_per_product = max(15, TARGET_DATASET_SIZE // len(unique_products))
    
    print(f"[2/3] Generating dataset (Final Balancing Strategy)...")

    for prod in unique_products:
        # === LOGIC: DOWNSAMPLE THE NOISE ===
        if prod['category'] in HIGH_BOOST:
            n_rows = base_rows_per_product * 15  # 15x for Tablets
        elif prod['category'] in MEDIUM_BOOST:
            n_rows = base_rows_per_product * 3   # 3x for Phones
        else:
            # DOWNSAMPLE ACCESSORIES to 20%
            # This prevents "HOCO" from overtaking the dataset
            n_rows = max(5, int(base_rows_per_product * 0.2)) 

        n_pos = int(n_rows * 0.5)
        
        # === POSITIVES ===
        base_positives = [
            prod['model'],                         
            prod['brand'],                         
            f"{prod['category']} {prod['brand']}", 
            prod['category'],                      
            prod['name']                           
        ]

        # Force keywords for rare items
        if prod['category'] == "Routeur_Modem":
            base_positives.extend(["wifi", "modem 4g", f"wifi {prod['brand']}", f"modem {prod['brand']}"])

        if prod['category'] == "Tablette":
             base_positives.extend(["tab", "tablette android", "tablette d-tech", "tablette 4g"])
        
        for _ in range(n_pos):
            base = random.choice(base_positives)
            if len(base.split()) > 4:
                query = base.lower()
            else:
                query = augment_query(base)

            dataset_rows.append({
                "product_id": prod['id'],
                "product_name": prod['name'],
                "category": prod['category'],
                "description": prod['model'],
                "price": prod['price'],
                "user_query": query,
                "relevance_label": 1 # MATCH
            })

        # === NEGATIVES ===
        n_neg = n_rows - n_pos
        
        for _ in range(n_neg):
            other = random.choice(unique_products)
            while other['id'] == prod['id']:
                other = random.choice(unique_products)
            
            # Smart Negative: Teach HOCO that it is NOT a Tablette
            if prod['category'] not in HIGH_BOOST and other['category'] in HIGH_BOOST:
                 # 100% chance to use the category name to strictly separate them
                 query_base = other['category'] 
            else:
                 query_base = other['name']

            if len(query_base.split()) > 4:
                query = query_base.lower()
            else:
                query = augment_query(query_base)
            
            dataset_rows.append({
                "product_id": prod['id'],
                "product_name": prod['name'],
                "category": prod['category'],
                "description": prod['model'],
                "price": prod['price'],
                "user_query": query,
                "relevance_label": 0 # NO MATCH
            })

    # --- STEP 3: SAVE ---
    random.shuffle(dataset_rows)
    
    headers = ["product_id", "product_name", "category", "description", "price", "user_query", "relevance_label"]
    
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(dataset_rows)

    print(f"[3/3] Done! Generated {len(dataset_rows)} PERFECTLY balanced training rows.")

if __name__ == "__main__":
    create_large_dataset()