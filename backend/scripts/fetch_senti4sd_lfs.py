"""Usage (from external/pySenti4SD/java): python fetch_senti4sd_lfs.py

Fetch Git LFS objects from collab-uniba/Senti4SD and verify SHA-256 (LFS) and MD5 (pySenti4SD's DVC)."""
import hashlib, json, sys, urllib.request
OBJECTS = {  # file: (lfs sha256, size, dvc md5)
    "NgramsExtraction.jar": ("c69e6ffc573bba79089a6a8e5836d503737d0717aefc7954797acc8a12de2bc6", 13568432, "d66bbaaa07739cfbd4cb8b94565d43f8"),
    "Senti4SD-fast.jar": ("9a5306f6667232248398c43718c1e71485f351209174e2ce4d3978dd22a7bd60", 61243352, "0ab05dae382556f265736d7a80d7b5e8"),
    "Senti4SD.jar": ("ee20aac22e50be1f0730142963abc108ef4e3f7a68656cae60bc7498f09df361", 51300173, "37c5228913ac9d2a76e6313023cd8160"),
    "dsm.bin": ("1db28b959bed55fd0ab377f608026503ad0920e8b1a383ff77be25caec279057", 835369209, "395cb470cacd6e584508d108945b67d3"),
}
req = urllib.request.Request(
    "https://github.com/collab-uniba/Senti4SD.git/info/lfs/objects/batch",
    data=json.dumps({"operation": "download", "transfers": ["basic"],
                     "objects": [{"oid": o, "size": s} for o, s, _ in OBJECTS.values()]}).encode(),
    headers={"Accept": "application/vnd.git-lfs+json", "Content-Type": "application/vnd.git-lfs+json"})
batch = json.load(urllib.request.urlopen(req))
hrefs = {o["oid"]: o["actions"]["download"]["href"] for o in batch["objects"] if "actions" in o}
errors = {o["oid"]: o.get("error") for o in batch["objects"] if "error" in o}
if errors: sys.exit(f"LFS errors: {errors}")
ok = True
for name, (oid, size, md5) in OBJECTS.items():
    sha, m, n = hashlib.sha256(), hashlib.md5(), 0
    with urllib.request.urlopen(hrefs[oid]) as r, open(name + ".part", "wb") as f:
        while chunk := r.read(1 << 20):
            f.write(chunk); sha.update(chunk); m.update(chunk); n += len(chunk)
    good = n == size and sha.hexdigest() == oid and m.hexdigest() == md5
    print(f"{name:22} {n:>11} bytes  sha256 {'OK' if sha.hexdigest()==oid else 'MISMATCH'}  md5 {'OK' if m.hexdigest()==md5 else 'MISMATCH'}")
    if good:
        import os; os.replace(name + ".part", name)
    ok &= good
sys.exit(0 if ok else 1)
