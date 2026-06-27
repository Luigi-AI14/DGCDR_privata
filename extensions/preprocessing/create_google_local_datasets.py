import json
import csv
import os
import datetime

workspace_dir = r"c:\Users\Gabriele\Desktop\Progetto_Narducci_Definitivo\DGCDR_privata"
business_path = os.path.join(workspace_dir, 'dataset_google_local', 'meta-California.json')
review_path = os.path.join(workspace_dir, 'dataset_google_local', 'review-California.json')
out_dir = os.path.join(workspace_dir, 'csv_datasets')

os.makedirs(out_dir, exist_ok=True)

restaurants_csv_path = os.path.join(out_dir, 'google_restaurants.csv')
hotels_csv_path = os.path.join(out_dir, 'google_hotels.csv')

# 1. Identify businesses
restaurant_ids = set()
hotel_ids = set()

print("Parsing business dataset...")
with open(business_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        gmap_id = data.get("gmap_id")
        categories = data.get("category")

        # Se il locale non ha categorie o non ha un ID valido, salta
        if not categories or not gmap_id:
            continue

        # Facciamo il check case-insensitive all'interno dell'array delle categorie
        for cat in categories:
            cat_lower = str(cat).lower()

            if "restaurant" in cat_lower:
                restaurant_ids.add(gmap_id)
                break  # Trovato, inutile leggere le altre categorie di questo locale

            if "hotel" in cat_lower or "lodging" in cat_lower or "resort" in cat_lower:
                hotel_ids.add(gmap_id)
                break

print(f"Found {len(restaurant_ids)} Restaurants and {len(hotel_ids)} Hotels.")


# --- 2. ESTRAZIONE RECENSIONI E SCRITTURA CSV ---
print("Parsing delle recensioni e generazione dei CSV...")

with open(review_path, 'r', encoding='utf-8') as fin, \
     open(restaurants_csv_path, 'w', newline='', encoding='utf-8') as fout_rest, \
     open(hotels_csv_path, 'w', newline='', encoding='utf-8') as fout_hotel:

    rest_writer = csv.writer(fout_rest)
    hotel_writer = csv.writer(fout_hotel)

    headers = ["user_id", "item_id", "rating", "timestamp"]
    rest_writer.writerow(headers)
    hotel_writer.writerow(headers)

    rest_count = 0
    hotel_count = 0

    for i, line in enumerate(fin):
        if i > 0 and i % 1000000 == 0:
            print(f"Elaborate {i} recensioni...")

        data = json.loads(line)
        gmap_id = data.get("gmap_id")

        # Verifica fulminea sui Set in memoria O(1)
        is_rest = gmap_id in restaurant_ids
        is_hot = gmap_id in hotel_ids

        if is_rest or is_hot:
            user_id = data.get("user_id")
            rating = data.get("rating")
            timestamp = data.get("time")  # È già un intero a 13 cifre (millisecondi)

            # Controllo di sicurezza anti-dati corrotti
            if user_id and rating is not None and timestamp:
                row = [user_id, gmap_id, rating, timestamp]

                if is_rest:
                    rest_writer.writerow(row)
                    rest_count += 1
                if is_hot:
                    hotel_writer.writerow(row)
                    hotel_count += 1

print(
    f"Finito! Scritte {rest_count} recensioni Ristoranti e {hotel_count} recensioni Hotel."
)