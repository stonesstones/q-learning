# Q-learning
## Code Structure
`train.py` - main function for training and evaluate agent

`models.py` - Qtable

`logger.py` - Logger
## Setup
```bash
pip3 install tensorboard dm-control torch torchvision torchaudio opencv-python
```
## For training
```bash
python train.py
```

## tensorboard
学習状況を見る
```bash
tensorboard --logdir ./logs
```

## Args
train.pyのEnvConfigの値を適宜変更する。

注意点: shouderの角度の分割数は0.5*num_digitized
```python
class EnvConfig:
    domain: str = "double_pendulum"     # single_pendulumか、double_pendulum
    task: str = "swingup"               # swingupしかない
    num_digitized: int = 16             # １つの値を何分割するか
    num_action: int = 2                 # actionの分割数
    state_size: int = num_digitized**4  # 状態の総数
    gamma: float = 0.99                 # 割引率
    alpha: float = 0.5                  # qtableの更新率
    max_episode: int = int(10e7)        # 学習するエピソードの総数
    episode_length: int = 400           # １エピソードの長さ
    should_log_model: int = 10000       # 何ステップごとにqtableを保存するか
    should_log_scalar: int = 100        # 何ステップごとにrewardをlogに加えるか
    should_log_video: int = 10000       # 何ステップごとに現在のqtableを評価し動画にするか
    restore: bool = False               # 学習済みのqtableを使用するか
    restore_file: str = "Qtable.npy"    # 学習済みのqtableのパス
    video_length: int = 400             # 評価するときのエピソードの長さ
    logdir: str = "./logs/" + str(time.strftime("%Y-%m-%d-%H-%M-%S")) + "/" #logを残すディレクトリ
```

## Simulator
`sim/acrobot.py`のinitialize_episode関数を変更することで初期値を設定できる。（デフォルトではelbowの角度が$`[-0.5\pi, 0.5\pi]`$の間のランダムな値で、shouderの角度が$`0`$）

`sim/acrobot.xml`を変えることで、物体の重さ、長さ、ジョイントのダンピングや摩擦を変更できる。

具体的には、下の表に対応する数値を下のコードの対応する位置に入れる。

| パラメータ | upper arm | lower arm |
| :---: | :---: | :---: |
| mass | m_u | m_l |
| length | l_u | l_l |
| size(diameter) | s_u | s_l |
| damping | d_u | d_l |
| friction | f_u | f_l |

```xml
<body name="upper_arm" pos="0 0 2">
    <joint name="shoulder" type="hinge" axis="0 0 1" damping="d_u" frictionloss="f_u"/>
    <geom name="upper_arm_decoration" material="decoration" type="cylinder" fromto="0 0 -.01 0 0 .01" size="s_u" mass="0"/>
    <geom name="upper_arm" fromto="0 0 0 l_u 0 0" size="s_u" material="self" rgba="1 1 0 0.6" mass="m_u"/>
    <body name="lower_arm" pos="l_u 0 0">
        <joint name="elbow" type="hinge" axis="1 0 0" damping="d_l" frictionloss="f_l"/>
        <geom name="lower_arm" fromto="0 0 0 0 0 l_l" size="s_l" material="self" rgba="1 0 1 0.6" mass="m_l"/>
        <site name="tip" pos="0 0 -l_l" size="0.01"/>
        <geom name="lower_arm_decoration" material="decoration" type="cylinder" fromto="-.01 0 0 .01 0 0" size="s_l" mass="0"/>
    </body>
</body>
```