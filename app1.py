import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np

from flask import Flask, request, jsonify
import os
import re
import tempfile
import mysql.connector as mysql
from dotenv import load_dotenv
import easyocr

load_dotenv()

app = Flask(__name__)

# MySQL config from .env
mysql_config = {
    "host": os.getenv("MYSQL_HOST"),
    "user": os.getenv("MYSQL_USER"),
    "password": os.getenv("MYSQL_PASSWORD"),
    "database": os.getenv("MYSQL_DATABASE")
}

# Initialize EasyOCR reader once globally (English only)
reader = easyocr.Reader(['en'], download_enabled=False)


def extract_boxes(image_path, conf_threshold=0.6):
    # Use EasyOCR to read image text with bounding boxes and confidence
    results = reader.readtext(image_path, detail=1)  # detail=1 returns list of (bbox, text, conf)
    
    boxes = []
    for (bbox, text, conf) in results:
        if conf >= conf_threshold and text.strip():
            x_center = sum([point[0] for point in bbox]) / 4
            y_center = sum([point[1] for point in bbox]) / 4
            boxes.append({
                'text': text.strip(),
                'conf': conf,
                'box': bbox,
                'x': x_center,
                'y': y_center
            })
    return boxes


def group_by_rows(boxes, y_thresh=15):
    boxes.sort(key=lambda b: b["y"])
    rows = []
    current_row = []
    last_y = -1000

    for box in boxes:
        if abs(box["y"] - last_y) > y_thresh:
            if current_row:
                rows.append(sorted(current_row, key=lambda b: b["x"]))
            current_row = [box]
        else:
            current_row.append(box)
        last_y = box["y"]

    if current_row:
        rows.append(sorted(current_row, key=lambda b: b["x"]))
    return rows


def detect_price(text):
    return re.search(r'(\u20B9|Rs\.?)?\s?\d{1,4}([.,]\d{1,2})?', text)


def is_valid_item(text):
    if not text or len(text.strip()) <= 2 or (text.strip().isupper() and len(text.strip()) <= 3):
        return False
    noise_keywords = {
        'am', 'pm', 'yo', 'l', 't', 'a', 'b', '/', '-', '|', ':', '.', ',', '–', '—', '_', '(', ')',
        'daily', 'only', 'each', 'per', 'day', 'week', 'month', 'with', 'served', 'includes',
        'available', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun', 'timings', 'timing',
        'from', 'at', 'till', 'until', 'for', 'special', 'offer', 'extra', 'add-on',
        'optional', 'combo', 'set', 'option', 'mrp', 'gst', 'inclusive', 'exclusive',
        'taxes', 'inc.', 'excl.', 'incl.'
    }
    clean = re.sub(r'[^\w]', '', text).lower()
    return clean not in noise_keywords and re.search(r'[a-zA-Z]', text)

import shutil

@app.route('/extract', methods=['POST'])
def extract():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    # Step 1: Save to temp file and close it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    # Step 2: Copy to a new working file (Windows-safe)
    working_path = tmp_path + "_copy.jpg"
    shutil.copyfile(tmp_path, working_path)

    # Step 3: Now we can safely process the copy
    boxes = extract_boxes(working_path)
    rows = group_by_rows(boxes)

    result = []
    for row in rows:
        texts = [box['text'] for box in row if is_valid_item(box['text'])]
        if texts:
            result.append(" ".join(texts))

    # Step 4: Clean up both files
    try:
        os.unlink(tmp_path)
    except Exception as e:
        print("Temp file cleanup failed:", e)

    try:
        os.unlink(working_path)
    except Exception as e:
        print("Working file cleanup failed:", e)

    return jsonify({'data': result})


def assign_categories(rows):
    categorized_rows = []
    current_category = "Uncategorized"
    for row in rows:
        texts = [box["text"] for box in row]
        full_line = " ".join(texts).strip()
        if detect_price(full_line):
            for box in row:
                box["category"] = current_category
            categorized_rows.append(row)
            continue
        words = full_line.split()
        uppercase_words = sum(1 for w in words if w.isupper() or w.istitle())
        if uppercase_words >= max(1, len(words) // 2) and len(full_line) <= 35 and len(words) <= 4:
            current_category = full_line
            continue
        for box in row:
            box["category"] = current_category
        categorized_rows.append(row)
    return categorized_rows


def parse_rows_to_menu(categorized_rows, image_name="unknown"):
    menu = []
    last_entry = None
    for row in categorized_rows:
        row.sort(key=lambda b: b["x"])
        line = " ".join([b["text"] for b in row])
        cat = row[0].get("category", "Uncategorized")
        price_matches = list(re.finditer(r'(₹|Rs\.?)?\s?\d{1,5}([.,]\d{1,2})?', line))
        if price_matches:
            items, prices = [], []
            start = 0
            for match in price_matches:
                price = match.group().strip()
                item_chunk = line[start:match.start()].strip(" -–—|,")
                item_texts = re.split(r'\s{2,}|,|/| - | \| |\. ', item_chunk)
                for it in item_texts:
                    it = re.sub(r'\(.*?\)', '', it).strip()
                    if is_valid_item(it):
                        items.append(it)
                        prices.append(price)
                start = match.end()
            for i in range(min(len(items), len(prices))):
                entry = {"image": image_name, "category": cat, "item": items[i], "price": prices[i], "description": ""}
                last_entry = entry
                menu.append(entry)
        elif last_entry and cat == last_entry["category"]:
            last_entry["description"] += " " + line
    return menu


def insert_into_mysql(data, config, vendor_id):
    try:
        conn = mysql.connect(**config)
        cursor = conn.cursor()
        query = """
        INSERT INTO menu_or_services (category, item_or_service, price, description, vendor_id, image_path)
        VALUES (%s, %s, %s, %s, %s, %s)
        """
        for entry in data:
            cursor.execute(query, (
                entry["category"], entry["item"], entry["price"],
                entry["description"], vendor_id, entry["image"]
            ))
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except mysql.Error as err:
        print("MySQL Error:", err)
        return False

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    elif hasattr(obj, 'tolist'):  # for numpy arrays or scalars
        return obj.tolist()
    elif isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    else:
        return str(obj)


def convert_np(obj):
    if isinstance(obj, dict):
        return {k: convert_np(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np(i) for i in obj]
    elif isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    else:
        return obj

@app.route("/upload", methods=["POST"])
def upload():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    image = request.files['image']

    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        image.save(tmp.name)
        tmp_path = tmp.name

    try:
        boxes = extract_boxes(tmp_path)
        # (your MySQL insertion or processing here...)

        return jsonify({'status': 'success', 'boxes': convert_np(boxes)})

    finally:
        # Ensure the file is closed and removed
        try:
            os.remove(tmp_path)
        except Exception as e:
            print(f"Failed to delete temp file {tmp_path}: {e}")


if __name__ == "__main__":
    # app.run(debug=True)
   
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)

