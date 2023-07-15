import utility
# noinspection PyUnresolvedReferences
import sys
import argparse
from useroracle import UserStrategy, UserOracle
# noinspection PyUnresolvedReferences
import state
# noinspection PyUnresolvedReferences
import useroracle
import mcts
from state import State, MenuState, UserState
import ray
from copy import deepcopy
import os

# Setup command-line arguments and options
parser = argparse.ArgumentParser()  # 创建一个解析对象
# 向该对象中添加你要关注的命令行参数和选项 default - 当参数未在命令行中出现时使用的默认值 action - 命令行遇到参数时的动作
# 参数解释 choices - 用来选择输入参数的范围。例如choice = [1, 5, 10], 表示输入参数只能为1,5 或10   help - 用来描述这个选项的作用
parser.add_argument("--menu", "-m", help="Input menu name", default="menu_5items.txt")
parser.add_argument("--history", "-H", help="Click frequency file name", default="history_5items.csv")  # 用户点击频率历史
parser.add_argument("--associations", "-a", help="Association list file name", default="associations_5items.txt")  # 关联列表
parser.add_argument("--strategy", "-s", help="User search strategy", default="average", choices=["serial","forage","recall","average"])
parser.add_argument("--time", "-t", type=int, help="time budget", default=5000)
parser.add_argument("--iterations", "-i", type=int, help="num iterations", default=200) # 迭代次数，默认为200
parser.add_argument("--depth", "-d", type=int, help="maximum depth", default=5)  # 最大深度，默认为5
parser.add_argument("--nopp",help="disable parallelisation", action='store_true')  # 禁用并行化计算的标志
parser.add_argument("--pp", type=int, help="number of parallel processes", default=10)  # 并行处理的进程数，默认为10
parser.add_argument("--usenetwork", "-nn", help="Use neural network", action='store_true')
parser.add_argument("--valuenet","-vn",help="Value network name")
parser.add_argument("--case", "-c", help="Use case e.g. 5items, 10items, toy (combination of menu, assoc, history)")
parser.add_argument("--objective", "-O", help="Objective to use", choices = ["average","optimistic","conservative"], default="average")  # 使用的目标函数，平均；最优；保守

args = parser.parse_args()  # 解析添加的参数

use_network = True if args.usenetwork else False

# Value network model names
if args.menu == "menu_7items.txt":
    vn_name = "value_network_7items.h5"
elif args.menu == "menu_5items.txt":
    vn_name = "value_network_5items.h5"
elif args.menu == "menu_10items.txt":
    vn_name = "value_network_10items.h5"
elif args.menu == "menu_15items.txt":
    vn_name = "value_network_15items.h5"
else: 
    vn_name: None
    use_network = False

#Objective function to be used; default is average 目标函数默认选择average
objective = args.objective.upper()

# Change PWD to main directory
pwd = os.chdir(os.path.dirname(__file__)) # 将当前工作目录更改为脚本所在文件的父目录 避免在代码中使用绝对路径

# Set up the menu instance
currentmenu = utility.load_menu("./input/" + args.menu)  # load menu items from text file 调用utility.py 里面的函数来导入菜单列表
# freqdist：item频率列表  total_clicks: 用户点击menu里item的总数 history:点击历史记录列表 元素为[item名字,在menu里的下标]
freqdist, total_clicks, history = utility.load_click_distribution(currentmenu, "./input/" + args.history) # load from user history (CSV file)
associations = utility.load_associations(currentmenu,"./input/" + args.associations) # load assocation matrix from text file

# If --case is included in CLI arguments
if args.case is not None: # case 赋值 5item/10item/15item 替换参数
    currentmenu = utility.load_menu("./input/menu_" + args.case + ".txt")
    freqdist, total_clicks, history = utility.load_click_distribution(currentmenu, "./input/history_" + args.case + ".csv")
    associations = utility.load_associations(currentmenu,"./input/associations_" + args.case + ".txt")
    vn_name = "value_network_" + args.case + ".h5"

# If different objective function is specified
strategy = UserStrategy.AVERAGE
if args.strategy == "serial":
    strategy = UserStrategy.SERIAL
elif args.strategy == "forage":
    strategy = UserStrategy.FORAGE
elif args.strategy == "recall":
    strategy = UserStrategy.RECALL

# MCTS search parameters 控制MCTS搜索的行为和性能 maxdepth 决定搜索的深度限制，timebudget决定搜索的时间限制，iteration_limit决定搜索的迭代次数限制
maxdepth = args.depth
timebudget = args.time
iteration_limit = args.iterations

weights = [0.25,0.5,0.25] #  默认 average Weights for the 3 models 权重值用于控制三个模型在菜单适应性规划过程中的影响力。根据不同的搜索策略，可以通过调整权重来改变不同模型的权衡和重要性，以满足特定的需求或优化目标

if strategy == UserStrategy.SERIAL:
    weights = [1.0, 0.0, 0.0]
elif strategy == UserStrategy.FORAGE:
    weights = [0.0, 1.0, 0.0]
elif strategy == UserStrategy.RECALL:
    weights = [0.0, 0.0, 1.0]

# Intialise the root state using the input menu, associations, and user history
menu_state = MenuState(currentmenu, associations)  # 菜单状态由当前菜单列表以及菜单item关联列表构成
user_state = UserState(freqdist, total_clicks, history)  # 用户状态由 freqdist：用户点击菜单item频率列表 total_clicks: 用户点击menu里item的总数 history:点击历史记录列表 元素为[item名字,在menu里的下标]

root_state = State(menu_state,user_state, exposed=True)  # 初始化状态根节点
my_oracle = UserOracle(maxdepth, associations=menu_state.associations)  #用户运行模型 由最大深度 菜单关联列表 决定
completion_times = my_oracle.get_individual_rewards(root_state)[1]  # Initial completion time for current menu 三种策略搜索时间列表
avg_time = sum([a * b for a, b in zip(weights, completion_times)])  # 计算加权平均搜索时间（三种策略都考虑）
parallelised = False if args.nopp else True  # 是否并行


# Start the planner
ray.init()  # ray.init()来初始化Ray库，启动了一个计算节点
# 打印原始菜单、平均选择时间、用户兴趣（归一化的点击频率分布）和关联性信息
print(f"Planning started. Strategy: {strategy}. Parallelisation: {parallelised}. Neural Network: {use_network}.")
print(f"Original menu: {menu_state.simplified_menu()}. Average selection time: {round(avg_time,2)} seconds")
print(f"User Interest (normalised): {freqdist}")
print(f"Associations: {associations}")

# Execute the MCTS planner and return sequence of adaptations
@ray.remote
def step(state, oracle, weights, objective, use_network, network_name, timebudget):  # 执行一步，并保存了步骤的结果
    results = []
    original_times = oracle.get_individual_rewards(state)[1]  # 获取这一步 使用各策略搜索时间列表 [new_serial_time, new_forage_time, new_recall_time]
    tree = mcts.mcts(oracle, weights, objective, use_network, network_name, time_limit=timebudget)
    node = None
    while not oracle.is_terminal(state):  # 没到结束条件 获取在此节点可选择的最佳adaptation 对应子节点 以及奖励（基于三种模型）
        _, best_child, _, _ = tree.search(state, node) # search returns selected (best) adaptation, child state, avg rewards
        node = best_child
        state = best_child.state
        [rewards, times] = oracle.get_individual_rewards(state)  # 选择这个子节点 各搜索策略的奖励列表 以及时间列表
        if objective == "AVERAGE":  # 三种搜索策略加权平均
            avg_reward = sum([a*b for a,b in zip(weights, rewards)])  # Take average reward
            avg_time = sum([a * b for a, b in zip(weights, times)])
            avg_original_time = sum([a*b for a,b in zip(weights,original_times)]) # average selection time for the original design
        elif objective == "OPTIMISTIC":  # 奖励选择三种搜索策略奖励最大的 搜索时间选择最小的
            avg_reward = max(rewards)  # Take best reward
            avg_time = min(times)
            avg_original_time = min(original_times)
        elif objective == "CONSERVATIVE":  # 奖励选择三种搜索策略奖励最小的 搜索时间选择最大的
            avg_reward = min(rewards)  # Take minimum; add penalty if negative
            avg_time = max(times)
            avg_original_time = max(original_times)
            
        #avg_reward = sum([a * b for a, b in zip(weights, rewards)])
        #avg_time = sum([a * b for a, b in zip(weights, times)])
        if avg_time > avg_original_time and state.exposed:   # 如果没有时间缩短 并且现在暴露给用户标志为真 将标志设为假
            exposed = False  # Heuristic. There must be a time improvement to show the menu
        else: exposed = state.exposed
        results.append([state.menu_state.simplified_menu(), state.depth, exposed, round(avg_original_time,2), round(avg_time,2), round(avg_reward,2)])
    return avg_reward, results

if not parallelised:
    result = step(root_state,my_oracle,weights, objective, use_network, vn_name, timebudget)
    bestmenu = result[1]
    # Get results and save output
    print("\nPlanning completed.\n\n[[Menu], Step #, Is Exposed, Original Avg Time, Final Avg Time, Reward]")
    for step in bestmenu:
        print(step)
        if step[2]: utility.save_menu(step[0], "output/adaptedmenu" + str(step[1]) + ".txt")  # 每个深度保存一个菜单文件
elif parallelised:  # Create and execute multiple instances 代码创建多个并行实例，并使用 Ray 框架来运行这些实例
    parallel_instances = args.pp  # Number of parallel instances 并行处理进程数
    state_copies = [deepcopy(root_state)] * parallel_instances  # Create copies
    result_ids = []
    for i in range(parallel_instances):
        statecopy = state_copies[i]
        result_ids.append(step.remote(statecopy, my_oracle, weights, objective, use_network, vn_name, timebudget))
    # 并行执行多个函数调用，并在需要时获取结果
    results = ray.get(result_ids)  # Use ray to run instances
    bestresult = float('-inf')  # 初始化最优结果和最优菜单
    bestmenu = menu_state.simplified_menu()

    # # Get best result from parallel threads
    for result in results:
        if result[0] > bestresult:
            bestresult = result[0] + 0.0
            bestmenu = result[1]

    # Get results and save output
    print("\nPlanning completed.\n\n[[Menu], Step #, Is Exposed, Original Avg Time, Final Avg Time, Reward]")
    for step in bestmenu:
        print(step)
        if step[2]: utility.save_menu(step[0], "output/adaptedmenu" + str(step[1]) + ".txt")  # 每个深度保存一个菜单文件