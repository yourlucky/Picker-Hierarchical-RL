
1. Load/visualize antbullet agent
```bash
$ python 1_loader.py
```
> change policy in line  18
> policy locatation example '1.sub_policy/PPO/ac'

2. Draw sub-policy graph
```bash
$ python3 0_plot_rl.py
```
3. Picker agent run
```bash
$ python3 picker_train_test.py
```

4. Train Sub-policy 

## Running
```bash
$ python ppo_train.py # for PPO
$ python sac_train.py # for SAC
$ python trpo_train.py # for TRPO
```