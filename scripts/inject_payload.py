import json, sys, copy

job_id  = sys.argv[1]
payload = json.loads(sys.argv[2])

with open("notebook/worker.ipynb", "r") as f:
    nb = json.load(f)

# Find the "# CONFIG" cell and overwrite it
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if "# CONFIG" in src:
        cell["source"] = [
            "# CONFIG — auto-injected by GitHub Actions\n",
            f"JOB_ID = {json.dumps(job_id)}\n",
            f"PAYLOAD = {json.dumps(payload)}\n",
        ]
        break

with open("notebook/worker.ipynb", "w") as f:
    json.dump(nb, f, indent=1)

print(f"Injected job_id={job_id} into worker.ipynb")
