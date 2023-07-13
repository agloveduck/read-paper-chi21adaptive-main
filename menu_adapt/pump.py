import  useroracle
import utility
import sys
import random
import mcts
from state import State, MenuState, UserState
import argparse
import numpy as np
import time

# 生成训练数据

# 使用argparse模块创建一个解析器
parser = argparse.ArgumentParser()
parser.add_argument("--menu", "-m", help="Input menu name", default="menu_5items.txt")
parser.add_argument("--history", "-H", help="Click frequency file name", default="history3.csv")
parser.add_argument("--associations", "-a", help="Association list file name", default="associations_5items.txt")
parser.add_argument("--inputdir", "-i", help="Input directory path", default="./input")
parser.add_argument("--menuscount", "-mn", type=int, help="number of unique menus", default=50)
parser.add_argument("--usercount", "-un", type = int, help = "number of user histories", default = 4)
parser.add_argument("--adaptationcount", "-an", type=int, help="number of adaptations", default=50)
parser.add_argument("--timebudget", "-t", type=int, help="time budget", default=12000)
parser.add_argument("--maxdepth", "-d", type=int, help="maximum depth", default=8)
parser.add_argument("--pumptype", "-p", help="Pump Type (PN/VN)", choices=["PN","VN"], default="VN")
args = parser.parse_args()

def simplify_menu(menu): # 函数用于简化菜单，去除重复的分隔符，并确保菜单以分隔符开头和结尾
    separator = "----"
    simplified_menu = []
    for i in range (0,len(menu)):
        if menu[i] != separator: 
            simplified_menu.append(menu[i])
            continue
        if menu[i] == separator and len(simplified_menu)>0:
            if simplified_menu[-1] == separator: continue
            simplified_menu.append(menu[i])
    if simplified_menu[0] == separator:
            del simplified_menu[0]
    if simplified_menu[-1] == separator:
            del simplified_menu[-1]
    num_additional_separators = len(menu) - len(simplified_menu)
    for _ in range(num_additional_separators): simplified_menu.append(separator)
    return simplified_menu

weights = [0.2,0.7,0.1] # Weights for the 3 models
use_network = False  # 默认不使用价值网络进行奖励估计
network = None

# 通过使用MCTS算法，根据当前状态和用户模型，在给定时间内搜索最佳的菜单调整方案，并返回相关的输出。
# State 当前状态（菜单/用户） oracle用户模型，预测任务完成时间和计算奖励 time_budget 时间限制
def pump(state, oracle, time_budget):  # 返回元组（） 元组包含 平均奖励列表 最优子节点的菜单 是否暴露给用户
    treesearch = mcts.mcts(oracle, weights, objective = "AVERAGE", use_network = use_network, network_name = network, limit_type='time', time_limit = time_budget)
    # adaptation = treesearch.search(initial_state=state)
    # state = state.take_adaptation(adaptation)
    # rewards = oracle.get_individual_rewards(state)[0]
    _, best_child, avg_rewards, _ = treesearch.search(initial_state=state)  # 调用mcts search 获得最优子节点 平均奖励列表
    return avg_rewards, best_child.state.menu_state.menu, best_child.state.exposed

def policy_pump(state,oracle,time_budget):
    treesearch = mcts.mcts(oracle, weights, objective = "AVERAGE", use_network = use_network, network_name = network, limit_type='time', time_limit = time_budget)
    _, _, _, probabilities = treesearch.search(initial_state=state)
    return probabilities
    
# Returns association matrix for a menu using the associations dictionary
def get_association_matrix(menu, associations): # 根据关联字典生成菜单的关联矩阵
    association_matrix = []
    for k in range (0, len(menu)):
        if menu[k] in associations:
            for l in range (0, len(menu)):
                if menu[l] in associations[menu[k]]:
                    association_matrix.append(1.0)
                else:
                    association_matrix.append(0.0)
        else:
            for l in range (0, len(menu)):
                association_matrix.append(0.0)
    return association_matrix

# Returns sorted frequencies list for a menu using the frequency dictionary
def get_sorted_frequencies(menu,frequency): # 根据频率字典生成菜单的频率列表
    separator = "----"
    sorted_frequencies = []
    for k in range (0, len(menu)):
        if menu[k] == separator:
            sorted_frequencies.append(0.0)
        else:
            sorted_frequencies.append(frequency[menu[k]])
    return sorted_frequencies

def generate_history(menu): #生成随机的历史记录列表 生成符合Zipf分布的随机数
    menu_items = list(filter(("----").__ne__, menu))
    probabilities = []
    while(1):
        probabilities = []
        shape = 1.5 # For zipf distribution shape是Zipf分布的形状参数，控制分布的偏斜程度。较大的shape值会使分布更加偏斜，而较小的shape值会使分布更加平坦
        size = len(menu_items)
        zipf_dist = np.random.zipf(shape,size) # 齐夫定律
        probabilities = [i/sum(zipf_dist) for i in zipf_dist]
        if max(probabilities) < 0.8: break
    clicks = random.choices(menu_items,probabilities,k=40)
    history = []
    for click in clicks:
        history.append([click, menu_items.index(click)])
    return(history) # History of size 40
    



# pump controls
number_of_unique_menus = args.menuscount
number_of_users = args.usercount
number_of_adaptations = args.adaptationcount
time_budget = args.timebudget
maxdepth = args.maxdepth

# task instance
currentmenu = utility.load_menu("./input/" + args.menu)
associations = utility.load_associations(currentmenu,"input/" + args.associations)
separators = len(associations.keys())
separator = "----"
print(currentmenu)
# initialization



training_data = set([])  # 创建一个空集合 存储训练数据
timestamp = time.strftime("%H%M%S")  #获取当前时间的时间戳
for i in range (0, number_of_unique_menus):
    print("Unique menu #: ",i+1)  # 调整到第几个菜单了
    my_menu_state = MenuState(currentmenu,associations)  # 初始化菜单状态
    random.shuffle(my_menu_state.menu)   # 对菜单进行随机打乱
    randomizedmenu = my_menu_state.menu  # 打乱后的菜单赋值给变量randomizedmenu
    print(randomizedmenu)  # 打印弄乱的菜单
    for j in range (0, number_of_users):
        history = generate_history(randomizedmenu)  # 随机生成的用户点击历史记录
        frequency,total_clicks = utility.get_frequencies(randomizedmenu, history)
        my_user_state = UserState(frequency, total_clicks,history)   # 使用频率、总点击数和随机历史记录初始化用户状态
        is_exposed = bool(random.getrandbits(1))  # 随机决定是否暴露给用户
        my_state = State(my_menu_state, my_user_state, exposed=is_exposed)  # 初始化状态
        my_oracle = useroracle.UserOracle(maxdepth,associations=my_state.menu_state.associations)  # 模拟用户操作
        old_menu = simplify_menu(randomizedmenu)
        frequency_old_menu = get_sorted_frequencies(old_menu,frequency)  # old_menu的频率列表
        associations_old_menu = get_association_matrix(old_menu,associations)  # old_menu的关联矩阵
        
        for k in range (0, number_of_adaptations):  # 生成训练数据，并将结果存储到文件中。
            if args.pumptype == "VN":   # 根据value network 策略 来进行菜单调整
                vn_results = pump(my_state, my_oracle, time_budget)  # 元组 （平均奖励列表 最优子节点的菜单 是否暴露给用户）
                if vn_results:
                    rewards = [round(i,5) for i in vn_results[0]]
                    new_menu = simplify_menu(vn_results[1])
                    exposed = vn_results[2]
                    frequency_new_menu = get_sorted_frequencies(new_menu,frequency)  # new_menu的频率列表
                    associations_new_menu = get_association_matrix(new_menu,associations)  # new_menu的关联矩阵
                    #  每一行数据如下 [serial,forage,recall][source_menu][source_frequencies][source_associations][target_menu][target_frequencies][target_associations][exposed]
                    observation =str(rewards) + str(old_menu) + str(frequency_old_menu) + str(associations_old_menu) + \
                                str(new_menu) + str(frequency_new_menu) + str(associations_new_menu) + str([int(exposed)])
                    training_data.add(observation)
                    with open('output/results_vn_' + timestamp + '.txt', 'a') as filehandle:
                        filehandle.write('%s\n'% observation)  # 创建一个新文件并写入观察数据的字符串表示形式
            elif args.pumptype == "PN":  # 根据policy network 策略 来进行菜单调整
                pn_results = policy_pump(my_state, my_oracle, time_budget)
                printable_results = {}
                if pn_results:
                    for key, value in pn_results.items():
                        printable_results[str(key)] = value

                        

                observation = str(old_menu) + str(frequency_old_menu) + str(associations_old_menu) + str([int(is_exposed)]) + str(printable_results)
                with open('output/results_pn_' + timestamp + '.txt', 'a') as filehandle:
                    filehandle.write('%s\n'% observation)



# with open('output/results' + time.strftime("%H%M%S") + '.txt', 'w') as filehandle:
#     for sample in training_data:
#         filehandle.write('%s\n' % sample)


