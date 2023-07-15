## 在MCTS的 ROLL-OUT 阶段 奖励估计 使用HCI预测模型（数据量比较小的情况 具体在菜单这个例子可能就是20item以内的情况）OR 神经网络（数据量较大情况）
### utility.py 从文件里读取数据加工成模型需要的数据
* load_menu (filename)导入 menu里的item列表
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
### mcts.py 使用蒙特卡洛树搜索进行规划 定义了TreeNode 四个阶段的函数等等
* 使用HCI预测模型进行奖励估计 调用函数random_policy(state, oracle)  该函数调用了useroracle.py的get_individual_rewards(self,state)函数
  这个模型的奖励是按用户兴趣加权的平均选择时间差异（previous_xxx_time - new_xxx_time）平均搜索时间 按照item点击频率加权
* 使用神经网络进行奖励估计 调用函数get_reward_predictions(self, node) 该函数调用了train好的value_network里面的predict_batch(self, data)函数
  [见value_network的model.py](./value_network/model.py)
  其中data为该节点的[source_menu,source_freq,source_assoc,target_menu,target_freq,target_assoc,[bool(exposed)]]信息
* 这两种方法都是返回累积奖励列表[reward_serial, reward_forage, reward_recall] 
* search(self, initial_state, initial_node = None)执行 MCTS 算法的过程，以找到最优adaptation 最优子节点 平均奖励列表 该节点的adaptation概率字典
### plan.py 
* 默认 5item 的 menu history association txt 就在input文件夹下 不使用value network进行奖励估计
* 


