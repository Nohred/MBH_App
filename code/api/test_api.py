import argparse
import time

import httpx


parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--file", required=True)
parser.add_argument("--model", choices=["model_a", "model_b"], required=True)
args = parser.parse_args()

start = time.perf_counter()
with open(args.file, "rb") as image_file:
    response = httpx.post(
        f"{args.url}/api/v1/predict",
        files={"file": (args.file, image_file, "application/gzip")},
        data={"model_id": args.model},
        timeout=None,
    )
elapsed = time.perf_counter() - start

print(f"status={response.status_code}")
print(f"elapsed_seconds={elapsed:.3f}")
print(f"content_type={response.headers.get('content-type')}")
if response.is_success:
    output_path = f"mask_{args.model}.nii.gz"
    with open(output_path, "wb") as output_file:
        output_file.write(response.content)
    print(f"saved={output_path}")
else:
    print(response.text)
