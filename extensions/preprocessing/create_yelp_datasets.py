import json
import csv
import os
import datetime

workspace_dir = r"c:\Users\Gabriele\Desktop\Progetto_Narducci_Definitivo\DGCDR_privata"
business_path = os.path.join(workspace_dir, 'dataset_yelp', 'yelp_academic_dataset_business.json')
review_path = os.path.join(workspace_dir, 'dataset_yelp', 'yelp_academic_dataset_review.json')
out_dir = os.path.join(workspace_dir, 'csv_datasets')

os.makedirs(out_dir, exist_ok=True)

restaurants_csv_path = os.path.join(out_dir, 'yelp_restaurants.csv')
hotels_csv_path = os.path.join(out_dir, 'yelp_hotels.csv')

# 1. Identify businesses
restaurant_ids = set()
hotel_ids = set()

print("Parsing business dataset...")
with open(business_path, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        business_id = data.get('business_id')
        categories = data.get('categories')
        
        if not categories:
            continue
            
        cat_list = [c.strip() for c in categories.split(',')]
        if 'Restaurants' in cat_list:
            restaurant_ids.add(business_id)
        if 'Hotels' in cat_list or 'Hotels & Travel' in cat_list:
            hotel_ids.add(business_id)

print(f"Found {len(restaurant_ids)} Restaurants and {len(hotel_ids)} Hotels.")

# 2. Extract reviews
print("Parsing reviews dataset and writing CSVs...")
with open(review_path, 'r', encoding='utf-8') as fin, \
     open(restaurants_csv_path, 'w', newline='', encoding='utf-8') as fout_rest, \
     open(hotels_csv_path, 'w', newline='', encoding='utf-8') as fout_hotel:
     
     rest_writer = csv.writer(fout_rest)
     hotel_writer = csv.writer(fout_hotel)
     
     headers = ['user_id', 'item_id', 'rating', 'timestamp']
     rest_writer.writerow(headers)
     hotel_writer.writerow(headers)
     
     rest_count = 0
     hotel_count = 0
     
     for i, line in enumerate(fin):
         if i > 0 and i % 1000000 == 0:
             print(f"Processed {i} reviews...")
             
         data = json.loads(line)
         b_id = data.get('business_id')
         
         if b_id in restaurant_ids or b_id in hotel_ids:
             user_id = data.get('user_id')
             rating = data.get('stars')
             date_str = data.get('date') # "2018-07-07 22:09:11"
             
             # Convert to unix timestamp in MILLISECONDS to match RecBole .inter format
             dt = datetime.datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S")
             timestamp = int(dt.timestamp() * 1000)
             
             row = [user_id, b_id, rating, timestamp]
             
             if b_id in restaurant_ids:
                 rest_writer.writerow(row)
                 rest_count += 1
             if b_id in hotel_ids:
                 hotel_writer.writerow(row)
                 hotel_count += 1

print(f"Done! Written {rest_count} reviews for Restaurants and {hotel_count} reviews for Hotels.")