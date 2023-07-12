# MCTS Implementation. During rollouts, the user oracle is used to predict rewards for adaptations

from __future__ import division, print_function
import time
import math
import random
import sys
import utility
import os
from useroracle import UserOracle
from copy import deepcopy
from adaptation import Adaptation
from state import AdaptationType
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'value_network'))  # 将模型文件的路径添加到Python解释器的搜索路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'policy_network'))
from value_network_model import ValueNetwork


# Rollout policy: random
def random_policy(state, oracle):
    rewards = [0.0,0.0,0.0]
    # if state.exposed: rewards = oracle.get_individual_rewards(state)[0]
    while not oracle.is_terminal(state):
        try:
            adaptation = random.choice(state.menu_state.possible_adaptations())
        except IndexError:
            raise Exception("Non-terminal state has no possible adaptations: " + str(state))
        state = state.take_adaptation(adaptation)
        if state.exposed:
            new_rewards = oracle.get_individual_rewards(state)[0]
            rewards = [a + b for a,b in zip(rewards, new_rewards)]                
    return rewards

# MCTS node
class TreeNode():  # 存储有关该节点的菜单状态、父节点、该节点的访问次数、访问获得的总奖励以及子节点的信息
    def __init__(self, state, parent):
        self.state = state  # Menu now 节点菜单的当前状态
        self.parent = parent  # 当前节点的父节点
        self.num_visits = 0  # For tracking n in UCT 该节点被访问的次数
        self.total_rewards = [0.0,0.0,0.0]  # For tracking q in UCT 三种模型对应的奖励列表
        self.children = {}  # 映射到子节点的字典  key:adaption 一个调整 value:子节点
        self.fully_expanded = False  # Is it expanded already? 是否已扩展

    def __str__(self):
        return str(self.state) + "," + str(self.total_rewards)  # 打印当前状态和总奖励

# MCTS tree
class mcts():
    def __init__(self, useroracle, weights, objective, use_network, network_name = None, limit_type = 'time', time_limit=None, num_iterations=None, exploration_const=1.0/math.sqrt(2),
                 rollout_policy=random_policy):
        
        self.oracle = useroracle  # User oracle used 用户策略模型
        self.objective = objective # Average, Conservative, or optimistic objective - used to compute total reward 三种计算总奖励的方法
        self.weights = weights  # Weights for combining the 3 strategies when using the "average" objective 使用平均方法时的各搜索加权权重
        self.time_limit = time_limit  # Time limit to search 搜索过程的时间限制
        self.limit_type = limit_type  # Type of computation budget 计算预算的类型，可以是“时间”或“迭代”
        self.num_iterations = num_iterations  # No. of iterations to run 运行搜索的最大迭代次数
        self.exploration_const = exploration_const  # Original exploration constant: 1 / math.sqrt(2) ：UCT（树的置信上限）公式中使用的探索常数
        self.rollout = rollout_policy  # Rollout policy used
        self.use_network = use_network  # 是否使用价值网络进行奖励估计
        if self.use_network and network_name:
            self.vn = ValueNetwork("networks/"+network_name)

        
    def __str__(self):   # 返回mcts树结构的字符串表示形式。它从根节点开始，递归遍历其所有子节点，将它们的字符串表示形式添加到输出中
        tree_str = str(self.root) + "\n"
        for child in self.root.children.values():
            tree_str += str(child) + "\n"
        return tree_str
    
    def execute_round(self):
        node = self.select_node(self.root)        
        if node is not self.root and self.use_network:
            rewards = self.get_reward_predictions(node)
        else: 
            rewards = self.rollout(node.state, self.oracle)
        self.backpropagate(node, rewards)

    def search(self, initial_state, initial_node = None):
        if initial_node: 
            self.root = initial_node
            self.root.parent = None
        else: self.root = TreeNode(initial_state, None)
        time_limit = time.time() + self.time_limit / 1000
        if self.limit_type == 'time':
            while time.time() < time_limit:
                self.execute_round()            
        elif self.limit_type == 'iterations':
            for _ in self.num_iterations:
                self.execute_round()

        adaptation_probability = self.get_adaptation_probabilities(self.root, 0.0)
        best_child = self.get_best_child(self.root, 0.0)
        best_adaptation = self.get_adaptation(self.root, best_child)
        avg_rewards = [x/best_child.num_visits for x in best_child.total_rewards]
        
        return best_adaptation, best_child, avg_rewards, adaptation_probability


    def get_reward_predictions(self, node):
        rewards = [0.0,0.0,0.0]
        if node.parent is not None:
            samples = []
            target_menu = node.state.menu_state.simplified_menu(trailing_separators=True)
            source_menu = node.parent.state.menu_state.simplified_menu(trailing_separators=True)
            target_state = node.state
            source_state = node.parent.state
            source_assoc = utility.get_association_matrix(source_menu, source_state.menu_state.associations)
            source_freq = utility.get_sorted_frequencies(source_menu, source_state.user_state.freqdist)
            target_assoc = utility.get_association_matrix(target_menu, target_state.menu_state.associations)
            target_freq = utility.get_sorted_frequencies(target_menu, target_state.user_state.freqdist)
            exposed = node.state.exposed
            samples.append([source_menu,source_freq,source_assoc,target_menu,target_freq,target_assoc,[bool(exposed)]])
            predictions = self.vn.predict_batch([samples[0]]) # Get predictions from value network (if usenetwork is true)
            rewards = predictions[0]
        return rewards

        
    def select_node(self, node):  # 根据UCT算法选择树中的节点 选择未被探索的
        # 子节点
        while not self.oracle.is_terminal(node.state):  # 判断是否到达最大深度
            if node.fully_expanded:  # 如果一个节点完全展开，意味着它的所有子节点都已被探索，它会选择UCT分数最高的子节点接着探索
                node = self.get_best_child(node, self.exploration_const)
            else:  # 该节点未完全扩展
                return self.expand(node)  # 扩展这个节点
        return node

    def expand(self, node):  # 扩展节点
        adaptations = node.state.menu_state.possible_adaptations()  #获取可能的调整列表  返回的列表元素格式为Adaptation([i, j, type, expose])
        #Always try the "do nothing path first" 首先考虑什么都不做 ？ 为什么 可不可以调整
        if adaptations[-1] not in node.children.keys():  # 扩展节点子节点的调整列表没有“什么都不做”这个调整 创建一个什么都不做的子节点
            adaptation = adaptations[-1]
            newNode = TreeNode(node.state.take_adaptation(adaptation), node)  #通过对菜单进行“do nothing”的调整创建一个新的节点（新state 本node作为父节点）
            node.children[adaptation] = newNode  #该节点的子节点字典 {[0,0,AdaptationType.NONE,True]：newNode}
            return newNode  # 返回一个新节点

        random.shuffle(adaptations)  # 打乱调整列表有利于探索不同路径并提高 MCTS 算法的有效性
        for  adaptation in adaptations:
            if adaptation not in node.children.keys():
                newNode = TreeNode(node.state.take_adaptation(adaptation), node)  # 通过对菜单进行一个adaptation，创建一个新节点
                node.children[adaptation] = newNode  # 该节点的子节点字典 {[adaptation]：newNode}
                if len(adaptations) == len(node.children) or self.oracle.is_terminal(newNode.state):
                    node.fully_expanded = True  # 该节点是否被探索完的条件：1.已经利用所有可能的adaption来创建子节点 / 2.新节点到达最大限制深度
                return newNode
        raise Exception("Ouch! Should never reach here")

    def backpropagate(self, node, rewards):
        while node is not None:
            node.num_visits += 1
            node.total_rewards = [a+b for a,b in zip(node.total_rewards,rewards)]
            node = node.parent

    # Pick best child as next state
    def get_best_child(self, node, exploration_const):  # 从给定的父节点中选择最佳子节点
        best_value = float("-inf")
        best_node = None
        # return argmax(customFunction(node, frequencies, associations))
        children = list(node.children.values())  # 子节点列表
        random.shuffle(children)
        for child in children:
            # node value using UCT
            total_reward = self.compute_reward(child.total_rewards)
            node_value = total_reward/child.num_visits + exploration_const * math.sqrt(math.log(node.num_visits) / child.num_visits)
            
            if node_value > best_value:
                best_value = node_value
                best_node = child

        return best_node

    def compute_reward(self,total_rewards):
        if self.objective == "AVERAGE":
            total_reward = sum([a*b for a,b in zip(self.weights, total_rewards)]) # Take average reward 
        elif self.objective == "OPTIMISTIC":
            total_reward = max(total_rewards) # Take best reward
        elif self.objective == "CONSERVATIVE":
            total_reward = min(total_rewards) if min(total_rewards) >= 0 else min(total_rewards)*2 # Take minimum; add penalty if negative
        return total_reward


    def get_adaptation(self, root, best_child):
        for adaptation, node in root.children.items():
            if node is best_child:
                return adaptation
                
    def get_adaptation_probabilities(self, node, exploration_const):
        if node.children == 0: return None
        # Transition probability for children. Dict. Key = adaptation; Value = probability
        probability = {a:0.0 for a in node.state.menu_state.possible_adaptations()}
        for adaptation,child in node.children.items():
            probability[adaptation] = child.num_visits/node.num_visits
        return probability
        
       
    def get_best_adaptation(self, root):
        best_num_visits = 0
        best_results = {}
        for adaptation,child in root.children.items():
            if child.num_visits > best_num_visits:
              best_num_visits = child.num_visits
              best_results = {adaptation:child}
            elif child.num_visits == best_num_visits:
              best_num_visits = child.num_visits
              best_results[adaptation] = child
        
        best_adaptation, best_child = random.choice(list(best_results.items())) 
              
        return best_adaptation, best_child