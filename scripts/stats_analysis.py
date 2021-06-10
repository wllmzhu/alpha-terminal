import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt

BIN_SIZE = 10

stats_path = sys.argv[1]
model_ids = list(map(lambda x: int(os.path.splitext(x)[0]), os.listdir(stats_path)))
model_ids.sort()

stats = []
for id in model_ids:
	model_path = os.path.join(stats_path, f'{id}.json')	
	with open(model_path) as fp:
		stats.append(json.load(fp))

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def plot_sequence(seq, caption, agg_func, separate_win_lose=True):
	moving_seq = [agg_func(chunk) for chunk in chunks(seq, BIN_SIZE)]
	x = game_indices[::BIN_SIZE]

	if separate_win_lose:
		win_seq = [s for i, s in enumerate(seq) if stats[i]['winner'] == 1]
		moving_win_seq = [agg_func(chunk) for chunk in chunks(win_seq, BIN_SIZE)]
		x_win = win_indices[::BIN_SIZE]

		lose_seq = [s for i, s in enumerate(seq) if stats[i]['winner'] == 2]
		moving_lose_seq = [agg_func(chunk) for chunk in chunks(lose_seq, BIN_SIZE)]
		x_lose = lose_indices[::BIN_SIZE]

	plt.figure()
	plt.plot(x, moving_seq)
	if separate_win_lose:
		plt.plot(x_win, moving_win_seq, label='win')
		plt.plot(x_lose, moving_lose_seq, label='lose')
	plt.title(caption)
	plt.legend()
	plt.show()	

def get_win_rate(winner_stream):
	win_count = winner_stream.count(1)
	return win_count / len(winner_stream)

print("# of games: ", len(model_ids))
game_indices = model_ids
win_indices = [model_id for i, model_id in enumerate(model_ids) if stats[i]['winner'] == 1]
lose_indices = [model_id for i, model_id in enumerate(model_ids) if stats[i]['winner'] == 2]

# win rate
winner_stream = [s['winner'] for s in stats]
print("global win rate: ", get_win_rate(winner_stream))
plot_sequence(winner_stream, caption=f'{BIN_SIZE} games moving win rates', agg_func=get_win_rate, separate_win_lose=False)

# policy gradient loss
pg_loss = [s['policy_gradient_loss'] for s in stats]
plot_sequence(pg_loss, caption=f'{BIN_SIZE} policy gradient loss moving average', agg_func=np.mean)

# episode length
epi_lens = [s['episode_length'] for s in stats]
print(f"average episode length {np.mean(epi_lens)}")
plot_sequence(epi_lens, caption=f'{BIN_SIZE} games episode length moving average', agg_func=np.mean)

# action length
action_lens_cum = [s['action_length_cumulative'] for s in stats]
plot_sequence(action_lens_cum, caption=f'{BIN_SIZE} games cumulative action length moving average', agg_func=np.mean)
action_lens_mean = [s['action_length_mean'] for s in stats]
plot_sequence(action_lens_mean, caption=f'{BIN_SIZE} games mean action length moving average', agg_func=np.mean)
action_lens_std = [s['action_length_std'] for s in stats]
plot_sequence(action_lens_std, caption=f'{BIN_SIZE} games action length std moving average', agg_func=np.mean)
action_lens_max = [s['action_length_max'] for s in stats]
plot_sequence(action_lens_max, caption=f'{BIN_SIZE} games max action length moving average', agg_func=np.mean)


# return
return_cum = [s['return_cumulative'] for s in stats]
plot_sequence(return_cum, caption=f'{BIN_SIZE} games cumulative return moving average', agg_func=np.mean)