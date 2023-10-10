# Alpha-Terminal

A self-contained framework to train and deploy Reinforcement Learning (Policy Gradient) agents to play the game of [Terminal](https://terminal.c1games.com/), developed on top of the [starter kit](https://github.com/correlation-one/C1GamesStarterKit). 

Read our [report](https://drive.google.com/file/d/10hCg3SpPaRgtiNvW73wukVeLP_rYjElo/view) of the final performance of the agent we trained (on a single CPU) that achieved strong amateur performance.

![demo](https://github.com/wllmzhu/alpha-terminal/assets/52590858/a2c13540-cc8b-4ff3-918d-685961756ba7)


# About the game: Terminal

Terminal is a two-player zero-sum tower-defense game played on a 28-by-28 board. Each player begins the game with 30 Health Points. The goal is to reduce the opponent’s Health Points to 0 by strategically sending mobile units to achieve touchdowns on the opponent’s edges. In the meanwhile, the player needs to build defense structures to protect her own edge from the opponent’s touchdown.

# Training

Our architecture is inspired by the LSTM policy of [AlphaStar](https://www.deepmind.com/blog/alphastar-mastering-the-real-time-strategy-game-starcraft-ii), which enables the agent to output multiple actions per turn. The agent is trained using vanilla Policy Gradient.

To train the LSTM agent, run
```bash
python-algo/run_and_learn.sh
```

# Deployment

To run the agent locally, run
```bash
python-algo/run.sh
```

To see the agent play live, upload the `python-algo` folder to the [Terminal](https://terminal.c1games.com/) website.
