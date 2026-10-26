import os
import io
import time
import json
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

DATA_DIR = './data/flickr30k'
IMG_DIR = os.path.join(DATA_DIR, 'images')
CAP_PATH = os.path.join(DATA_DIR, 'captions.json')
TARGET_TOTAL = 3000
BATCH_SIZE = 100

os.makedirs(IMG_DIR, exist_ok=True)

# Load existing pairs
if os.path.exists(CAP_PATH):
    with open(CAP_PATH, 'r') as f:
        existing_pairs = json.load(f)
else:
    existing_pairs = []

current_count = len(existing_pairs)
print(f"Current Flickr30k image-caption pairs count: {current_count}")

if current_count >= TARGET_TOTAL:
    print(f"Target of {TARGET_TOTAL} pairs already met.")
    exit(0)

# Make backup of original captions.json if not already backed up
backup_path = os.path.join(DATA_DIR, 'captions_600_backup.json')
if not os.path.exists(backup_path) and current_count == 600:
    shutil.copy(CAP_PATH, backup_path)
    print(f"Created backup of original captions to {backup_path}")

needed = TARGET_TOTAL - current_count
print(f"Downloading {needed} additional images & captions to reach {TARGET_TOTAL} total...")

def process_and_save(img_url, file_path):
    try:
        resp = requests.get(img_url, timeout=12)
        if resp.status_code == 200:
            im = Image.open(io.BytesIO(resp.content)).convert('RGB')
            w, h = im.size
            crop_size = min(w, h)
            left = (w - crop_size) // 2
            top = (h - crop_size) // 2
            im_cropped = im.crop((left, top, left + crop_size, top + crop_size))
            im_64 = im_cropped.resize((64, 64), Image.Resampling.LANCZOS)
            im_64.save(file_path, 'JPEG', quality=95)
            return True
    except Exception:
        pass
    return False

# Plan downloads:
# 1. split=test: offset 600 to 1000 (up to 400 rows)
# 2. split=train: offset 0 to 2000+ (as many as needed to reach 3000)

sources = [
    ('test', 600, 1000),
    ('train', 0, 5000)
]

all_pairs = list(existing_pairs)
target_idx = current_count
start_time = time.time()
saved_count = 0

for split_name, start_off, max_off in sources:
    if target_idx >= TARGET_TOTAL:
        break
    
    current_offset = start_off
    while current_offset < max_off and target_idx < TARGET_TOTAL:
        fetch_len = min(BATCH_SIZE, TARGET_TOTAL - target_idx, max_off - current_offset)
        url = f"https://datasets-server.huggingface.co/rows?dataset=clip-benchmark/wds_flickr30k&config=default&split={split_name}&offset={current_offset}&length={fetch_len}"
        
        try:
            r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
            if r.status_code != 200:
                print(f"Warning: Failed to fetch {split_name} at offset {current_offset}, status: {r.status_code}")
                time.sleep(2)
                current_offset += fetch_len
                continue
            
            data = r.json()
            rows = data.get('rows', [])
            if not rows:
                print(f"No more rows available in {split_name} at offset {current_offset}")
                break
            
            tasks = []
            pending_items = []
            
            for row in rows:
                if target_idx >= TARGET_TOTAL:
                    break
                row_data = row.get('row', {})
                jpg_info = row_data.get('jpg', {})
                src = jpg_info.get('src') if isinstance(jpg_info, dict) else None
                txt_raw = row_data.get('txt', '')
                
                # Pick the first non-empty caption line
                caption_lines = [l.strip() for l in txt_raw.split('\n') if l.strip()]
                caption = caption_lines[0] if caption_lines else "A scene from flickr."
                
                if not src:
                    continue
                
                fname = f"{target_idx:05d}.jpg"
                fpath = os.path.join(IMG_DIR, fname)
                pending_items.append((src, fpath, fname, caption))
                target_idx += 1
            
            with ThreadPoolExecutor(max_workers=20) as executor:
                future_to_item = {
                    executor.submit(process_and_save, item[0], item[1]): item
                    for item in pending_items
                }
                for future in as_completed(future_to_item):
                    item = future_to_item[future]
                    if future.result():
                        saved_count += 1
                        all_pairs.append({'image': item[2], 'caption': item[3]})
                    else:
                        print(f"Failed to process image for {item[2]}")
            
            current_offset += fetch_len
            elapsed = time.time() - start_time
            print(f"Progress: {len(all_pairs)}/{TARGET_TOTAL} pairs saved | Elapsed: {elapsed:.1f}s")
            
        except Exception as e:
            print(f"Error at {split_name} offset {current_offset}: {e}")
            time.sleep(1)
            current_offset += fetch_len

# Sort all_pairs by image name
all_pairs.sort(key=lambda x: x['image'])

# Save updated captions.json
with open(CAP_PATH, 'w') as f:
    json.dump(all_pairs, f, indent=2)

final_img_count = len([f for f in os.listdir(IMG_DIR) if f.endswith('.jpg')])
print(f"\n==========================================")
print(f"COMPLETED! Total Flickr30k images: {final_img_count}")
print(f"Total Flickr30k image-caption pairs in {CAP_PATH}: {len(all_pairs)}")
print(f"Time taken: {time.time() - start_time:.1f}s")
print(f"==========================================")
