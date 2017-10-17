# Asynchronous_Breakout_v0

This is a tensorflow implementation of asynchronous one step deep Q learning algorithm 

## Demo

![Breakout_v0](/img/asynchronous_breakout_v0.gif)

Full paper here: https://arxiv.org/pdf/1602.01783v1.pdf


## Dependencies

* python 3.5
* tensorflow 1.1.0
* opencv 3.2.0
* openAI


## Usage

For Training Run:

```
$ python train_asynchronous_breakout.py
```

For Demo Run:

```
$ python play_asynchronous_breakout.py
```

## Credit

The Original implementation of this [project](https://github.com/Zeta36/Asynchronous-Methods-for-Deep-Reinforcement-Learning). I just add tensorboard 
functionalities and also modify it for openAI gym environment. The original project was gamming environment (pong & tetris) written in pygame library. 


