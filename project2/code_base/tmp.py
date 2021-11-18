import os
from tqdm import tqdm
import json

for root,dirs,files in os.walk('./imitation_learning/episodes/'):
    for file in tqdm(files):
        p=os.path.join(root,file)
        with open(p) as f:
            json_load = json.load(f)
        if not 'Toad Brigade' in json_load['info']['TeamNames']:
            os.remove(p)