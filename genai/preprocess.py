import yaml
import box
from modelutils import read_data
import os
import json
import pandas as pd

with open('../config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))

for file in os.listdir('../data/'):
    if not os.path.isfile(os.path.join('../data/', file)):
        continue
    filename = os.fsdecode(file)
    print(filename)
    df = pd.DataFrame()
    if '.txt' in file:
        df = read_data(os.path.join('../data/', filename))
        print(df.head())
    elif '.json' in file:
        with open(os.path.join('../data/', filename), 'r', encoding='utf8') as j:
            contents = json.loads(j.read())
        df = pd.json_normalize(contents['messages'])[['date', 'from', 'text']].rename(columns={'date': 'datetime',
                                                                                       'from': 'person',
                                                                                       'text': 'content'})
        df = df.loc[df.content != '']

    df['person'] = df['person'] + ': '
    df['datetime'] = df['datetime'] + '- '
    df = df.to_string(index=False, header=False).split('\n')
    df = [' '.join(ele.split()) for ele in df]
    df = '\n'.join(df)
    df = df.encode('utf8')
    with open(os.path.join(cfg.folder_location, 'processed_' + file), 'wb') as f:
        f.write(df)


