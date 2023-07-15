## 在MCTS的 ROLL-OUT 阶段 奖励估计 使用HCI预测模型（数据量比较小的情况 具体在菜单这个例子可能就是20item以内的情况）OR 神经网络（数据量较大情况）
### utility.py 从文件里读取数据加工成模型需要的数据
* load_menu (filename)导入 menu 返回item列表
* load_click_distribution (menu, filename, normalize = True) 从history.csv文件里导入数据 
  返回 freqdist：item频率列表  total_clicks: 用户点击menu里item的总数 history:点击历史记录列表 元素为[item名字,在menu里的下标]
* load_associations (menu, filename): 关联字典 key是item名字 value是与之关联的item列表,如果没有与之关联的 列表里就是item自身
### state.py 定义了MCTS中每一个节点状态 state由 menustate userstate 两部分构成
* menustate 由 menu, associations决定 
* possible_adaptations(self)返回此菜单状态下可能的调整的列表 考虑的adapt包括 swaps / moves / Swap groups / Move groups / do nothing
  列表元素： Adaptation([i, j, type, expose])
* adapt_menu(self, adaptation) 根据Adaptation([i, j, type, expose])调整菜单
* userstate 由 freqdist, total_clicks, history决定
* get_activations(self)计算每个菜单项的激活值 根据用户的点击历史和时间间隔计算得出的 返回一个嵌套字典
* update(self, menu, number_of_clicks = None)更新用户状态的方法
### adaption.py 定义一个adaption  i j 是菜单中两个位置，type指定adapt的类型(swap,move,group move...) expose 是一个布尔值，指定是否向用户公开adpation
### useroracle.py HCI预测模型 三种用户搜索模型的方法 根据论文5.2模型实现
* get_individual_rewards(self,state)  # 计算三种策略奖励 返回两个列表 第一个列表是各策略的奖励列表 第二个列表是各策略搜索时间列表
### mcts.py 使用蒙特卡洛树搜索进行规划 定义了TreeNode 四个阶段的函数等等
* 使用HCI预测模型进行奖励估计 调用函数random_policy(state, oracle)  该函数调用了useroracle.py的get_individual_rewards(self,state)函数
  这个模型的奖励是按用户兴趣加权的平均选择时间差异（previous_xxx_time - new_xxx_time）平均搜索时间 按照item点击频率加权
* 使用神经网络进行奖励估计 调用函数get_reward_predictions(self, node) 该函数调用了train好的value_network里面的predict_batch(self, data)函数
  [见value_network的model.py](./value_network/model.py)
  其中data为该节点的[source_menu,source_freq,source_assoc,target_menu,target_freq,target_assoc,[bool(exposed)]]信息
* 这两种方法都是返回累积奖励列表[reward_serial, reward_forage, reward_recall] 
* search(self, initial_state, initial_node = None)执行 MCTS 算法的过程，以找到最优adaptation 最优子节点 平均奖励列表 该节点的adaptation概率字典
### plan.py 
* 默认 5item 的 menu history association txt 就在input文件夹下 不使用value network进行奖励估计 使用需要-nn
* 默认策略使用AVERAGE choices=["serial","forage","recall","average"] 默认目标函数 AVERAGE
* 设置参数  控制MCTS搜索的行为和性能 maxdepth 决定搜索的深度限制，timebudget决定搜索的时间限制，iteration_limit决定搜索的迭代次数限制
* 默认三种搜索策略的权重 weights = [0.25,0.5,0.25]
* 初始化 menu_state user_state 构成蒙特卡洛树的root_state 初始化用户搜索模型my_oracle 计算根节点三种搜索策略奖励列表completion_time 
  计算加权后奖励值avg_time
* 默认parallelised
* step(state, oracle, weights, objective, use_network, network_name, timebudget)函数
  该函数创建一棵mcts树 mcts(oracle, weights, objective, use_network, network_name, time_limit=timebudget)
  从根节点开始调用mcts.py的search函数获取在此节点可选择的最佳adaptation 对应子节点 以及奖励和时间（基于三种模型）
  如果时间没有缩短 就不把该adaption暴露给用户 
  将[state.menu_state.simplified_menu(), state.depth, exposed, round(avg_original_time,2), round(avg_time,2), round(avg_reward,2)]加入结果列表
  直到到达停止条件 返回avg_reward, results列表
* 根据是否parallelised 具体操作有些区别 但都是调用step(root_state,my_oracle,weights, objective, use_network, vn_name, timebudget)函数 计算出results列表 并且每个深度保存一个文件
### 如果使用价值网络
* 首先使用pump.py 去生成一些训练数据 保存在output/results_vn_' + timestamp + '.txt'
  用5item数据初始化菜单状态 并对菜单进行随机打乱 使用zipfian分布模拟菜单使用情况 生成数据的每一行包括
[serial,forage,recall][source_menu][source_frequencies][source_associations][target_menu][target_frequencies][target_associations][exposed]
* 将数据喂给train.py 获得h5文件 即plan.py里使用的模型
