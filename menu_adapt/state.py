from copy import deepcopy
import operator
import utility
from enum import Enum
from adaptation import Adaptation
import math
class AdaptationType(Enum):
    NONE = 0
    MOVE = 1
    SWAP = 2
    GROUP_MOVE = 3
    GROUP_SWAP = 4
    ADD_SEP = 5
    REMOVE_SEP = 6
    MOVE_SEP = 7

    def __str__ (self):
        return self.name
    
    def __repr__(self):
        return self.name

class State():
    separator = "----"
    number_of_clicks = 20
    # Initialise the state
    def __init__(self, menu_state, user_state, previous_seen_state = None, depth = 0, exposed = False):
        self.user_state = user_state
        self.menu_state = menu_state
        self.previous_seen_state = previous_seen_state
        self.depth = depth
        self.exposed = exposed

    # Function called when an adaptation is made. The user state and menu state are updated accordingly
    def take_adaptation(self, adaptation, update_user = True):        
        new_state = deepcopy(self)
        new_state.depth += 1
        new_state.exposed = adaptation.expose
        if self.exposed: new_state.previous_seen_state = self

        # Simulate the next user session by adding clicks
        if self.exposed and update_user:
            new_state.user_state.update(menu = self.menu_state.menu, number_of_clicks = self.number_of_clicks)
        # Adapt the menu

        new_state.menu_state.menu = self.menu_state.adapt_menu(adaptation)
        
        return new_state


# Defines the menu - includes the list of menu items, and association list for each item. 
class MenuState():  # 定义菜单状态
    separator = "----"
    def __init__(self, menu, associations): # 创建菜单的状态对象，并对菜单进行初始化和分隔符的添加
        self.menu = menu
        self.associations = associations
        separatorsalready = menu.count(self.separator) # How many separators we have?
        # max_separators = int(min(math.ceil(len(self.menu)/1.5), 8))
        max_separators = 4  # 根据菜单的长度动态确定最大分隔符数量
        if separatorsalready < max_separators:
            for _ in range (separatorsalready, max_separators): # Append the remaining of separators
                self.menu.append(self.separator)


    def __str__(self):
        return str(self.simplified_menu())

    def __repr__(self):
        return str(self.simplified_menu())
    
    # Returns list of all adaptations that are feasible from the menu state.
    def possible_adaptations(self): # 返回此菜单状态下可能的调整的列表 考虑的adapt包括 swaps / moves / Swap groups / Move groups / do nothing
        
        possibleadaptations = []
        seen_menus = [self.simplified_menu()]  # 去除多余分隔符的菜单 seen_menus 可以起到记录已经访问的菜单的作用，并用于避免不必要的重复计算和评估
        max_distance = 3

        simple_menu = list(filter(("----").__ne__, self.menu)) # menu without separators # 没有分隔符的菜单
            
        # swaps 两个嵌套的循环遍历菜单中的每个菜单项，以找到可以进行交换的菜单项
        for i in range (0, len(self.menu)):
            for j in range (i+1, len(self.menu)):
                if self.menu[i] == self.separator and self.menu[j] == self.separator: continue  # 相邻两item都是分隔符 无需交换
                if len(simple_menu)>10:
                    if abs(i-j)>max_distance and self.menu[i] is not self.separator:  # 检查是否超过了允许的最大交换距离
                        continue # Limit max swap distance
                test_adaptation = Adaptation([i,j,AdaptationType.SWAP, True])  # adapt类型是swap 向用户公布调整
                adapted_menu = MenuState(self.adapt_menu(test_adaptation), self.associations)   # 根据adapt类型调整后的菜单
                if (adapted_menu.simplified_menu() not in seen_menus): # 简化adapt后的menu 并检查是否在记录过往菜单的seen_menus列表里 该适应性生成了一个不同于之前出现过的菜单
                    seen_menus.append(adapted_menu.simplified_menu())  # 将adapt后的菜单加入已调整菜单列表
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.SWAP, True])) # 在可能的调整列表中加入 交换菜单中索引为 i 和 j 的菜单项，并设置这个adapt 用户可见或者不可见
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.SWAP, False]))

        # moves
        for i in range (0, len(self.menu)):
            for j in range (0, len(self.menu)):
                if i == j or (i == j+1): continue  # 避免移动菜单项到相同的位置或相邻的位置，因为这样的移动没有意义
                if len(simple_menu) > 10:
                    if self.menu[i] != self.separator and abs(j-i) > max_distance:  # 限制大型菜单中移动菜单项的最大距离
                        continue # Limit max move distance for large menus
                if self.menu[i] == self.separator:
                    if j != len(self.menu)-1 and (self.menu[j] == self.separator or self.menu[j-1] == self.separator):  # 避免将分隔符移动到其他位置或将菜单项移动到分隔符之前的位置
                        continue   
                test_adaptation = Adaptation([i,j,AdaptationType.MOVE, True])  #  adapt类型是move 向用户公布调整
                adapted_menu = MenuState(self.adapt_menu(test_adaptation), self.associations)
                if (adapted_menu.simplified_menu() not in seen_menus):
                    seen_menus.append(adapted_menu.simplified_menu())                     
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.MOVE, True])) # 在可能的调整列表中加入 移动菜单中索引为i的菜单项到位置j，并设置这个adapt 用户可见或者不可见
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.MOVE, False]))
                
        # group adaptations 使用分隔符对字符串进行分割，将其拆分为一个包含各个分组的列表 groups [[],[]...]
        menu_string = ";".join(self.menu)
        groups = menu_string.split(self.separator)
        groups = list(filter((";").__ne__, groups))
        groups = list(filter(("").__ne__, groups))
        # Swap groups
        for i in range (0, len(groups)):
            for j in range (i+1, len(groups)):
                test_adaptation = Adaptation([i,j,AdaptationType.GROUP_SWAP, True])  #  adapt类型是swap groups 向用户公布调整
                adapted_menu = MenuState(self.adapt_menu(test_adaptation), self.associations)
                if (adapted_menu.simplified_menu() not in seen_menus):
                    seen_menus.append(adapted_menu.simplified_menu())
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.GROUP_SWAP, True])) # 在可能的调整列表中加入 交换菜单中索引为i，j的分组，并设置这个adapt 用户可见或者不可见
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.GROUP_SWAP, False]))
        # Move groups
        for i in range (0, len(groups)):
            for j in range (0, len(groups)):
                if i == j or (i == j+1): continue  # 避免移动相同或相邻的位置分组
                test_adaptation = Adaptation([i,j,AdaptationType.GROUP_MOVE, True])
                adapted_menu = MenuState(self.adapt_menu(test_adaptation), self.associations)
                if (adapted_menu.simplified_menu() not in seen_menus):
                    seen_menus.append(adapted_menu.simplified_menu())
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.GROUP_MOVE, True])) # 在可能的调整列表中加入 移动菜单中索引为i的分组到位置j，并设置这个adapt 用户可见或者不可见
                    possibleadaptations.append(Adaptation([i,j,AdaptationType.GROUP_MOVE, False]))
        
        # do nothing, show menu
        possibleadaptations.append(Adaptation([0,0,AdaptationType.NONE,True]))  # 在可能的调整列表加入 什么都不做的情况
        return possibleadaptations  # 返回的列表元素格式为Adaptation([i, j, type, expose])

    # Function to modify the menu by making an adaptation.
    def adapt_menu(self, adaptation):  # 根据adpat类型调整菜单
        new_menu = self.menu.copy() # 新菜单为当前菜单的副本
        if adaptation.type == AdaptationType.SWAP:  # swap 交换 i j 位置
            new_menu[adaptation.i], new_menu[adaptation.j] = new_menu[adaptation.j], new_menu[adaptation.i]
        elif adaptation.type == AdaptationType.MOVE:  # 移动 new_menu 中索引为 adaptation.i 的菜单项删除，并将其插入到索引为 adaptation.j 的位置
            del new_menu[adaptation.i]
            new_menu.insert(adaptation.j, self.menu[adaptation.i])
        elif adaptation.type == AdaptationType.GROUP_SWAP or adaptation.type == AdaptationType.GROUP_MOVE:
            menu_string = ";".join(new_menu)  # 将菜单转换为字符串形式，并使用分隔符进行分割
            groups = menu_string.split(self.separator)
            groups = list(filter((";").__ne__, groups))
            groups = list(filter(("").__ne__, groups))
            if adaptation.type == AdaptationType.GROUP_SWAP: # 交换第i,j两组
                groups[adaptation.i],groups[adaptation.j] = groups[adaptation.j], groups[adaptation.i]
            elif adaptation.type == AdaptationType.GROUP_MOVE: # 移动第i组到 位置j
                original_groups = groups.copy()
                del groups[adaptation.i]
                groups.insert(adaptation.j, original_groups[adaptation.i])
            groups_string = ";----;".join(groups)
            new_menu = groups_string.split(";")
            new_menu = list(filter("".__ne__,new_menu))
            missing_separators = len(self.menu) - len(new_menu)
            for _ in range (0, missing_separators): # Append the remaining of separators
                new_menu.append(self.separator)
        return new_menu

    # Returns a simplified representation of the menu by ignoring redundant/unnecessary separators
    def simplified_menu(self, trailing_separators = True):  # 返回一个简化表示的菜单，忽略多余/不必要的分隔符
        simplified_menu = []  # 空列表 simplified_menu 用于存储简化后的菜单
        for i in range (0,len(self.menu)):
            if self.menu[i] != self.separator: 
                simplified_menu.append(self.menu[i])
                continue
            if self.menu[i] == self.separator and len(simplified_menu)>0:
                if simplified_menu[-1] == self.separator: continue  # 最后一个也是分隔符 忽略 不要重复
                simplified_menu.append(self.menu[i])

        if simplified_menu[0] == self.separator:
                del simplified_menu[0]
        if simplified_menu[-1] == self.separator:
                del simplified_menu[-1]   
        if trailing_separators:  # 默认情况下保留尾部的分隔符
            old_length = len(self.menu)
            new_length = len(simplified_menu)
            sep_to_add = old_length - new_length
            for _ in range (sep_to_add): # Append the remaining of separators
                simplified_menu.append(self.separator)

        return simplified_menu
            
# Defines the user's interest and expertise
class UserState():  # 定义用户状态类
    def __init__(self, freqdist, total_clicks, history):  # 初始化
        self.freqdist = freqdist  # Normalised click frequency distribution (position-independent) 字典，freqdist 表示每个菜单项的点击频率
        self.total_clicks = total_clicks  # 总点击次数
        self.history = history  # 点击历史记录，是一个列表，记录了用户点击的菜单项及其位置
        item_history = [row[0] for row in self.history]  # 从历史记录里提取出用户点击的菜单项列表 可能有重复项

        self.recall_practice = {}  # Count of clicks at last-seen position (resets when position changes) 每个菜单项在最后一次出现位置的点击次数字典
        self.activations = self.get_activations()  # 每个菜单项的激活值
        for key,_ in self.freqdist.items():
            self.recall_practice[key] = item_history.count(key)
        if int(total_clicks) != len(self.history): print("HMM something wrong with the history")
        # for i in range(0, total_clicks):
        #     item = history[i][0]
        #     position = history[i][1]


    # For each item, returns a dictionary of activations.
    def get_activations(self): # 激活值是根据用户的点击历史和时间间隔计算得出的
        activations = {} # Activation per target per location 存储每个菜单项在每个位置的激活值
        duration_between_clicks = 20.0 # Wait time between two clicks 两次点击之间的等待时间
        session_interval = 50.0 # Wait time between 2 sessions 两个会话之间的等待时间
        session_click_length = 20 # Clicks per session 每个会话中的点击次数
        total_sessions = math.ceil(self.total_clicks/session_click_length) # Number of sessions so far 当前总共的会话数
        for i in range(0, int(self.total_clicks)):
            session = math.ceil((i+1)/session_click_length) # Session index
            item = self.history[i][0]  # 当前点击的菜单项
            position = self.history[i][1]  # 当前点击的菜单项所处位置
            if item not in activations.keys(): activations[item] = {position:0} # Item has not been seen yet. Add to dictionary 初始化其对应位置的激活值为0
            if position not in activations[item].keys(): activations[item][position] = 0 # Item not seen at this position yet. Add to item's dictionary
            # 在计算激活值时，越早点击的菜单项具有更高的激活值
            # 点击时间与当前时间之间的时间差 = 两次点击之间的等待时间 * 剩余点击次数 + 剩余会话次数 * 会话间等待时间
            time_difference = duration_between_clicks*(self.total_clicks - i) + (total_sessions - session)*session_interval # Difference between time now and time of click
            activations[item][position] += pow(time_difference, -0.5)  # 时间差的倒数的平方作为激活值的增量
        return activations  # 嵌套字典 例子如下
    # {
#     item1: {
#         position1: activation_value1,
#         position2: activation_value2,
#         ...
#     },
#     item2: {
#         position1: activation_value3,
#         position2: activation_value4,
#         ...
#     },
#     ...
# }



    # Method to update user state when the time-step is incremented (after taking an adaptation)
    def update(self, menu, number_of_clicks = None):
        # num_clicks = len(self.history)
        #  if len(self.history) < number_of_clicks else number_of_clicks
        # First we add new clicks
        clicks_to_add = self.simple_history()[-number_of_clicks:]
        
        item_list = list(filter(("----").__ne__, menu)) # new menu without separator
        for click in clicks_to_add:
            self.history.append([click, item_list.index(click)])

        # Next we update user expertise
        self.update_freqdist(menu)
        self.activations = self.get_activations()

    # Update frequency distribution based on new history    
    def update_freqdist(self, menu, normalize = True):
        self.freqdist = {}
        for command in menu:
            if command != "----":
                self.freqdist[command] = 0
        
        for item in self.simple_history():
            if item == "": continue
            if item not in list(self.freqdist.keys()):
                self.freqdist[item] = 1.
            else:
                self.freqdist[item] += 1.

        self.total_clicks = sum(list(self.freqdist.values()))

        if normalize: 
            for command in list(self.freqdist.keys()):
                self.freqdist[command] = round(self.freqdist[command]/self.total_clicks, 3)

    # click history without timestamp
    def simple_history(self):
        return [row[0] for row in self.history]

    def __str__(self):
        return str([self.freqdist, self.activations, self.total_clicks])
    def __repr__(self):
        return str([self.freqdist, self.activations, self.total_clicks])
