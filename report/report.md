# miniAlphaGo程序报告

[toc]

## 1. 个人基本信息

- 学生姓名：徐震
- 学号：3180105504
- 课程：《人工智能》2020春夏学期
- 教师姓名：吴飞
- 项目名称：miniAlphaGo for Reversi



## 2. 项目基本信息

### 2.1 实验基本要求

- 使用 **『最小最大搜索』**、**『Alpha-Beta 剪枝搜索』** 或 **『蒙特卡洛树搜索算法』** 实现 miniAlphaGo for Reversi（三种算法择一即可）。
- 使用 Python 语言。
- 算法部分需要自己实现，不要使用现成的包、工具或者接口。
- 提交程序报告,请在本地编辑并命名为**『程序报告.docx』**后，上传到左侧文件列表中。

### 2.2 实现与测试环境

- Python 3.7.4 (default, Aug  9 2019, 18:34:13) [MSC v.1915 64 bit (AMD64)] :: Anaconda, Inc. on win32
- Intel(R) Core(TM) i7-9759H CPU @ 2.60GHz  2.59 GHz
- Installed Memory: 16.0 GB (15.9 GB usable)
- Microsoft Windows [Version 10.0.18363.720]

### 2.3 使用/修改/添加API

已有的`board.py`，`game.py`以及`human_player`，`random_player`实现：

- 略，已经在课程作业中被详细介绍

AIPlayer用到的API：

- `board.py`
  - `get_legal_actions`
  - `_move`
  - `backpropagation`
  - `board_num`

测试过程中修改/添加的API：

- `game.py`

  - `run`：修改的是计时器，原程序使用`now`进行计时，不准确，甚至出现负数，我们测试中使用了`time.pref_counter`，来进行更高精度的性能测试

    ```python
    ...
    start_time = time.perf_counter()
    ...
    end_time = time.perf_counter()
    ...
    ```

    测试过程中遇到单子超过60s的性能测试需求，因此修改了`func_timeout`参数以进行测试

    ```python
    action = func_timeout(6000, self.current_player.get_move, kwargs={'board': self.board})
    # 第一个参数的单位是秒
    ```

  - `run_quite`：是我们为了方便大批量测试添加的API，只打印有用的log，有利于一目了然的观察性能信息，相对于原本的`run`，修改的地方莫过于注释掉`board.display`等语句

- `main.py`

  - 仿照`main.ipynb`中的最后一个代码块进行修改，方便本地测试，调试与性能验证
  - 有两种可能的使用模式：
    - 循环模式：每次游戏结束后给用户修改最大深度等超参数的机会，并按用户意愿继续进行或停止比赛
    - 测试模式：按照一定深度进行测试，调用`run_quite`API来保留重要的log信息

## 3. 项目原理

项目实现过程中用到了：

1. Min-Max游戏树搜索
2. Alpha-Beta剪枝搜索，递归式的函数重用
3. History-Table历史表辅助搜索
4. 基于棋盘权重，行动力，稳定子等的评估函数
5. 与游戏进度相关的动态评估能力
6. `multiprocess`提供的多核辅助搜索
7. `numpy`提供的向量化高速运算

注意到在项目中我们实现了单核，多核两种不同的搜索加速方式，两者的主要区别在于历史表，下面的原理解析中主要以单核版本为依据，最后说明多核版本的实现方式

### 3.1 核心思想：Min-Max游戏树搜索

我们采用了Min-Max搜索而不是蒙特卡洛树搜索作为主要实现方式的原因如下：

1. Min-Max相对容易实现与拓展，有更大的优化空间
2. 本次项目没有用到GPU训练蒙特卡洛树搜索需要的各类评估网络，评估效果不一定足够优秀
3. 阅读相关论文后发现普通游戏树搜索往往比那些没有经过精心调教的采样式搜索效果好
4. 蒙特卡洛树搜索涉及到一些复杂的数学运算，难以利用`numpy`提供的向量化加速优势
5. 蒙特卡洛树搜索过程中需要随时对各种信息进行更新，无法保证线程安全性，很难做到多核CPU优化

Min-Max的原理与书中描述的基本相同，只不过我们采用了递归的形式来简化实现难度，下面粘贴伪代码

```python
if terminal_state:
    return self.evaluate(board), action
# If no terminal state is encountered, we should just enumerate on possible moves computed above
for move in moves:
    flipped = board._move(move, color)  # Make a move
    val = -self.alpha_beta(board, -beta, -alpha, oppo_color, depth - 1)[0]  # Recursively compute the reward
    board.backpropagation(move, flipped, color)  # Reverse the change made to gaming board for the next enumeration
    # Update current maximum reward value
    if val > max_val:
        max_val = val
        action = move
return max_val, action
```

### 3.2 Alpha-Beta剪枝搜索，递归式的函数调用

我们使用了Alpha-Beta剪枝搜索以加快搜索速度，减少不必要的查找搜索开销

采用了普通的Alpha-Beta剪枝的搜索树策略的时间复杂度为$O(\sqrt{w}^d)$也就是$O({w}^{\frac{d}{2}})$

可以明显加快搜索过程，根据一些分析，若是我们对查找的内容简单排序，就可以逼近上述时间复杂度（也就是我们下面用到的History-Table）

注意我们使用了递归，并且将Max Node与Min Node合并到一起，若要将Min-Max Node的代码合并，我们需要进行以下工作：

1. 将每次调用`alpha_beta`函数的返回值调转符号作为其父节点的值
2. 每次调用`alpha_beta`函数时调换alpha与beta的值，并且调整他们的符号
3. 每次实现剪枝的时候都将本节点当作Max节点来看，即为：
   1. 当取得的`max_val`比我们已有的alpha值大的时候，更新alpha值
   2. 当取得的`max_val`比我们已有的beta值大的时候，进行剪枝

```python
if terminal_state:
    return self.evaluate(board), action
# If no terminal state is encountered, we should just enumerate on possible moves computed above
for move in moves:
    flipped = board._move(move, color)  # Make a move
    val = -self.alpha_beta(board, -beta, -alpha, oppo_color, depth - 1)[0]  # Recursively compute the reward
    board.backpropagation(move, flipped, color)  # Reverse the change made to gaming board for the next enumeration
    # Update current maximum reward value
    if val > max_val:
        max_val = val
        action = move
        # Update current alpha value for alpha-beta pruning
        if max_val > alpha:
            if max_val >= beta:
                action = move
                # The other children of current node should not be checked anymore
                # and reward current best move since is leads to an alpha-beta pruning
                self.reward_move(board, action, color, global_depth, True)
                return max_val, action
            # Update
            alpha = max_val
return max_val, action
```

### 3.3 考虑多重因素的评估函数

我们对Terminal State（或者Max Depth State）的评估函数进行了多方面的调优和测试。在查询资料过程中，我们了解到多种评估策略

1. 棋盘权重：

   类似于围棋，黑白棋也有金角银边草肚皮的简单规律。
   
   我们更倾向于让走子偏向于角点。
   
   我们在实验过程中采用过的一些权重有
   
   |      |  A   |  B   |  C   |  D   | E    |  F   |  G   |  H   |
   | :--: | :--: | :--: | :--: | :--: | ---- | :--: | :--: | :--: |
   |  1   |      |      |      |      |      |      |      |      |
   |  2   |      |      |      |      |      |      |      |      |
   |  3   |      |      |      |      |      |      |      |      |
   |  4   |      |      |      |      |      |      |      |      |
   |  5   |      |      |      |      |      |      |      |      |
   |  6   |      |      |      |      |      |      |      |      |
   |  7   |      |      |      |      |      |      |      |      |
   |  8   |      |      |      |      |      |      |      |      |
   
   
   
2. 

