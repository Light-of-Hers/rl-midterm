# 强化学习期中

[TOC]

## 使用方式

```
usage: run_agent.py [-h] [-m MAP] [-r] [-e EPISODES]

optional arguments:
  -h, --help            show this help message and exit
  -m MAP, --map MAP     map name
  -r, --rule_based      whether using rule-based agent
  -e EPISODES, --episodes EPISODES
                        max episodes
```



## Rule Based Agent

### 使用示例

在项目的根目录：

```shell
python3 -m pysc2.bin.agent --agent ruled_agent.MoveBeacon --map MoveToBeacon --use_feature_units
```

或者直接用已有脚本：

```shell
./run_agent MoveToBeacon
```

注：这些agent要用到SC2Env的`feature_units`（本质上可以由`feature_screen`推出），因此使用时需要启用该选项。



### 组件设计

#### `Rule`

`Rule`类，表示执行用的规则，主要由两个部分组成：

+ condition：一个`Observation -> Bool`的函数，用于判断前置条件。
+ actions：一个元素类型为`Observation -> Action`的列表，表示动作序列。

#### `RuledAgent`

Rule-based Agent的基类，主要功能是：

+ 设定一个`rules`列表，表示该agent基于的规则。
+ 每个step开始时，检查动作队列：
  + 若空，顺序检查每个rule的condition，选取第一个匹配到的，将其actions设为动作队列。
  + 否则，从队列头取出一个动作执行。



### 策略简述

#### MoveToBeacon

+ 策略较为简单，在每个step移动到最近的beacon即可。

#### CollectMineralShards

+ 轮流选择两个枪兵(marine)，移动到离其最近的shard（同时记录前一个枪兵的目标shard，避免两者目标重复）。

#### FindAndDefeatZerglings

+ 虽然该minigame有考察move camera相关的操作，但实际上完全基于minimap进行操作即可。
+ 在minimap上设置若干个巡逻点（当前设置为靠近地图边角的四个点），组成一个可以让视野覆盖整个地图的巡逻路线。
+ 初始时全选己方的三个枪兵。
+ 若当前视野内存在小狗(zergling)，则攻击最近的一个。
+ 若当前视野内已经没有小狗，则按设定的巡逻路线巡逻。

#### DefeatRoaches

+ 初始全选己方枪兵。
+ 每个step选择血量最少的蟑螂(roach)进行攻击（若血量相等，则选择位置最靠上的攻击）。

#### DefeatZerglingsAndBanelings

+ 若毒爆虫(baneling)没被清完，则每个step仅派出一个枪兵去试图攻击小狗，以吸引毒爆虫自爆。
+ 毒爆虫清完后，全选己方枪兵，每个step选择血量最少的小狗攻击。

#### CollectMineralsAndGas

+ 初始时派农民(SCV)采矿，并训练农民。
+ 资源足够后，在靠近右侧矿点的位置再建一个基地(command-center)。
+ 然后开始训练更多的农民采矿，人口不够了就造补给站(supply-depot)。
  + 注：好像并不需要造补给站就可以吊打论文中的RL-Agent了。

#### BuildMarines

+ 硬编码若干个点作为补给站建造点（当前设了20个）、若干个点作为兵营(barracks)建造点（当前设了8个）。
+ 训练农民采矿（当前设置农民上限为20个）。
+ 矿够了就建兵营。
+ 兵营建到7~8个就可不再建了，开始爆兵。
+ 期间人口不够了就造补给站。



### 效果对比

|                             | Mean, Max (Rule-based) | Mean, Max (Worst in paper) | Mean, Max (Best in paper) |
| --------------------------- | ---------------------- | -------------------------- | ------------------------- |
| MoveToBeacon                | 26.34, 32              | 25, 33                     | 26, 45                    |
| CollectMineralShards        | 110.54, 126            | 96, 131                    | 104, 137                  |
| FindAndDefeatZerglings      | 46.68, 52              | 45, 56                     | 49, 59                    |
| DefeatRoaches               | 100.38, 355            | 98, 373                    | 101, 351                  |
| DefeatZerglingsAndBanelings | 114.53, 220            | 62, 251                    | 96, 444                   |
| CollectMineralsAndGas       | 4839.51, 5045          | 3351, 3995                 | 3978, 4130                |
| BuildMarines                | 133.0, 139             | < 1, 20                    | 6, 62                     |

注：BuildMarines由于reset后采光的矿不会恢复的BUG，只跑了3个episodes，其他都跑了300个episodes。





## RL Based Agent

### 方法选择

基于打表的q_learning

训练中用时序差分TD来更新q_table：

```python
q_table[prev_state][action] = 0.5 * q_table[prev_state][action] + 0.5 * (reward + 0.99 * max_q_next)
```

训练和测试时的动作选择都采用epsilon-greedy.

```python
epsilon -> action = np.argmax(q_table[curr_state][:]) 
1-epsilon -> action = np.random.randint(0, num_action)
```



### 具体设计

#### FindAndDefeatZerglings

+ states x actions = 4 x 2
  + state_1：视野内敌人的数量，数据分箱[-inf, 1, 2, 3, inf]
  + action_1：攻击最近的敌人
  + action_2：沿给定路线巡逻

+ reward：直接用pysc2库给出的obs.reward，获取每一步的即时收益



### 效果对比

|                        | Mean, Max (RL-based) | Mean, Max (Rule-based) | Mean, Max (Worst in paper) | Mean, Max (Best in paper) |
| ---------------------- | -------------------- | ---------------------- | :------------------------- | ------------------------- |
| FindAndDefeatZerglings | 46.4，54             | 46.68, 52              | 45, 56                     | 49, 59                    |

![test_100ep](https://raw.githubusercontent.com/Light-of-Hers/my-picbed/master/test_100ep.png)

episode：100



### 效果分析

将视野内敌人的数量作为状态，Rule-Based认为只要有敌人就都选择attack，没有敌人就scout，相当于它的策略如下：

|         | scout | attack |
| ------- | ----- | ------ |
| 0 zerg  | √     |        |
| 1 zerg  |       | √      |
| 2 zerg  |       | √      |
| ≥3 zerg |       | √      |

RL-Based学出的是类似的效果，q_table可以反映出来策略偏好：

| 15.55 | 18.69 |
| ----- | ----- |
| 20.44 | 19.22 |
| 21.27 | 18.72 |
| 21.39 | 20.30 |

只是注意到过分偏离路线的Attack会导致遍历地图的速度变慢，所以测试时选择epsilon-greedy而非greedy，保留了在视野里有敌人的情况下仍然前行的可能性，这是RL_Based在Max_score上比Rule_Based表现较好的原因。





