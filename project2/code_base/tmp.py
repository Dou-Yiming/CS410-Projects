import os
from tqdm import tqdm
import json

for root,dirs,files in os.walk('E:/Datasets/LUX_ai/DATA/top/top'):
    for file in tqdm(files):
        p=os.path.join(root,file)
        if 'episodes' in p:
            new_name=p.replace("episodes","")
            if not os.path.exists(new_name):
                os.rename(p,new_name)
            else:
                os.remove(p)
        # with open(p) as f:
        #     json_load = json.load(f)
        # if not 'Toad Brigade' in json_load['info']['TeamNames']:
        #     os.remove(p)