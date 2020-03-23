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

