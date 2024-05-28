import os
import time
import random
import json
from glob import glob

from smdbz.core import generate_smdbz, read_smdbz

BASE_DIR = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    fps = sorted(glob(os.path.join(BASE_DIR, "data/global/*.png")))
    # print(f"len(fps): {len(fps)}")
    generate_smdbz(fps, "./global.smdbz")

    t0 = time.perf_counter()
    smdbz = read_smdbz("./global.smdbz")
    t1 = time.perf_counter()
    print(f"Load smdbz cost {t1-t0} seconds")

    with open("./idxs.json") as f:
        idxs = json.load(f)

    t2 = time.perf_counter()
    for _ in range(1000):
        iy, ix = random.choice(idxs)
        # xx = smdbz.get(iy, ix)
        xx = smdbz[iy, ix]
        # xx = [array[iy, ix] for array in smdbz]
        # print(xx)

    t3 = time.perf_counter()
    print(f"Access cost {round((t3 - t2)*1000, 2)} us")

    with open("./value_mapping.json") as f:
        value_mapping = json.load(f)
    
    for idx in value_mapping:
        iy, ix = idx["idx"]
        assert smdbz[iy, ix] == idx["value"]
