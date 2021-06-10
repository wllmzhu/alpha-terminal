import os
import json
from time import sleep
import random
import requests

VERIFICATION = 'trust-store.pem'
SLEEP_TIME = 0.01

def get_json(file_name: str, api_url: str) -> dict:
    if os.path.exists(file_name):
        with open(file_name) as fp:
            return json.load(fp)
    else:
        sleep(SLEEP_TIME)
        res = requests.get(api_url, verify='trust-store.pem')
        while res.status_code != 200:
            sleep(SLEEP_TIME)
            res = requests.get(api_url, verify='trust-store.pem')
        resJson = res.json()
        with open(file_name, 'w') as fp:
            json.dump(resJson, fp)
        return resJson

# get all competitions
competitions = get_json(
    'competition.json', 
    'https://terminal.c1games.com/api/game/competition'
)
comp_ids = list(map(lambda x: x['id'], competitions['data']['competitions']))
random.shuffle(comp_ids)
os.makedirs('competitions', exist_ok=True)

# get all matches
replays = []
for comp_id in comp_ids:
    os.makedirs(f'competitions/{comp_id}', exist_ok=True)
    competition = get_json(
        f'competitions/{comp_id}.json',
        f'https://terminal.c1games.com/api/game/competition/{comp_id}/matches'
    )
    replay_ids = list(map(lambda x: (x['id'], comp_id), competition['data']['matches']))
    replays.extend(replay_ids)
random.shuffle(replays)

# get replays
for replay_id, comp_id in replays:
    if os.path.exists(f'competitions/{comp_id}/{replay_id}.replay'):
        print(f'REPLAY {comp_id}/{replay_id} ALREADY EXIST')
    else:
        sleep(SLEEP_TIME)
        replay_url = f'https://terminal.c1games.com/api/game/replayexpanded/{replay_id}'
        res = requests.get(replay_url, verify=VERIFICATION)
        print(f'REPLAY {comp_id}/{replay_id} STATUS {res.status_code}')
        for _ in range(5):
            if res.status_code in {200, 404}:
                break
            sleep(SLEEP_TIME)
            res = requests.get(replay_url, verify=VERIFICATION)
            print(f'REPLAY {comp_id}/{replay_id} STATUS {res.status_code}')

        if res.status_code == 200:
            with open(f'competitions/{comp_id}/{replay_id}.replay', 'w') as fp:
                fp.write(res.text)
