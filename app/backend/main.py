from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from typing import List, Optional
import json
import datetime
import os

app = FastAPI()

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data
def get_products():
    with open("./products.json", "r", encoding="utf-8") as f:
        raw = json.load(f)
        return raw["products"]

# Log access
def log_access(product_id: int, custom_id: Optional[str] = None, test_case: Optional[str] = None, product_data: dict = None):
    """Log user access to access.jsonl file"""
    log_entry = {
        "id": custom_id if custom_id else str(product_id),
        "product_id": product_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "access_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # If test_case parameter exists, add to log
    if test_case:
        log_entry["test_case"] = test_case
    
    # Append to access.jsonl file
    with open("./access.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

app.mount("/data", StaticFiles(directory="data"), name="data")

@app.get("/products")
def get_all_products():
    return {"products": get_products()}

@app.get("/products/{product_id}")
def get_product(product_id: int, request: Request, id: Optional[str] = Query(None), test_case: Optional[str] = Query(None)):
    print(test_case)
    product = None
    for p in get_products():
        if p["id"] == product_id:
            product = p
            break
    
    if not product:
        raise HTTPException(status_code=404, detail="Product not found")
    
    custom_id = id if id else str(product_id)
    log_access(product_id, custom_id, test_case)
    
    return product

@app.get("/products/search")
def search_products(q: str = Query(..., min_length=1)):
    results = [p for p in get_products() if q.lower() in p["title"].lower()]

    return {"products":  get_products()}
    # return {"products": results}

@app.get("/logs/access")
def get_access_logs():
    if not os.path.exists("./access.jsonl"):
        return {"logs": [], "message": "No access logs found"}
    
    logs = []
    try:
        with open("./access.jsonl", "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    logs.append(json.loads(line.strip()))
        return {"logs": logs, "total": len(logs)}
    except Exception as e:
        return {"error": str(e), "logs": []}

@app.delete("/logs/access")
def clear_access_logs():
    try:
        if os.path.exists("./access.jsonl"):
            os.remove("./access.jsonl")
        return {"message": "Access logs cleared successfully"}
    except Exception as e:
        return {"error": str(e)}
