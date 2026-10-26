import os
import io
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image

OUT_DIR = './data/celeba'
TARGET_TOTAL = 3000
START_OFFSET = 1000
BATCH_SIZE = 100

os.makedirs(OUT_DIR, exist_ok=True)

existing_files = [f for f in os.listdir(OUT_DIR) if f.endswith('.jpg')]
current_count = len(existing_files)
print(f"Current CelebA images count: {current_count}")

needed = TARGET_TOTAL - current_count
if needed <= 0:
    print(f"Target of {TARGET_TOTAL} images already met.")
    exit(0)

print(f"Downloading {needed} additional images to reach {TARGET_TOTAL} total...")

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
    except Exception as e:
        pass
    return False

current_offset = START_OFFSET
target_idx = current_count

start_time = time.time()
total_saved = 0

while target_idx < TARGET_TOTAL:
    fetch_len = min(BATCH_SIZE, TARGET_TOTAL - target_idx)
    url = f"https://datasets-server.huggingface.co/rows?dataset=nielsr/CelebA-faces&config=default&split=train&offset={current_offset}&length={fetch_len}"
    
    try:
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        if r.status_code != 200:
            print(f"Failed to fetch rows at offset {current_offset}, status: {r.status_code}")
            time.sleep(2)
            continue
        data = r.json()
        rows = data.get('rows', [])
        if not rows:
            print(f"No more rows available at offset {current_offset}")
            break
            
        tasks = []
        with ThreadPoolExecutor(max_workers=16) as executor:
            for row in rows:
                src = row['row']['image']['src']
                fpath = os.path.join(OUT_DIR, f"{target_idx:05d}.jpg")
                tasks.append(executor.submit(process_and_save, src, fpath))
                target_idx += 1
                if target_idx >= TARGET_TOTAL:
                    break
                    
            for future in as_completed(tasks):
                if future.result():
                    total_saved += 1

        current_offset += fetch_len
        print(f"Progress: {target_idx}/{TARGET_TOTAL} images (saved {total_saved} new images) | Elapsed: {time.time()-start_time:.1f}s")
        
    except Exception as e:
        print(f"Error at offset {current_offset}: {e}")
        time.sleep(1)

final_count = len([f for f in os.listdir(OUT_DIR) if f.endswith('.jpg')])
print(f"\nCompleted! Total CelebA images now in {OUT_DIR}: {final_count}")
