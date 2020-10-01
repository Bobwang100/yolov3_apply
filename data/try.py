import json

with open('./tile_data_n8/train/TR20191023009991.json', 'r') as f:
    json_d = json.load(f)
    print(json_d)