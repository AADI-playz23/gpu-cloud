import sys, os, json, requests, glob

job_id  = sys.argv[1]
out_dir = sys.argv[2]
db_url  = sys.argv[3]          # e.g. https://yoursite.rf.gd/api/result.php

# Collect all .json output files from the Kaggle kernel output directory
results = {}
for path in glob.glob(os.path.join(out_dir, "*.json")):
    key = os.path.basename(path).replace(".json", "")
    with open(path) as f:
        results[key] = json.load(f)

payload = {
    "job_id":  job_id,
    "status":  "complete",
    "results": results,
}

resp = requests.post(db_url, json=payload, timeout=30)
resp.raise_for_status()
print(f"Result pushed for job {job_id} — HTTP {resp.status_code}")
