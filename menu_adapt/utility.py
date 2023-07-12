import csv
#For fasttext word embedding
# import fasttext
# .util
#For word2vec embeddings
# from gensim.models import KeyedVectors
#To compute cosine similarity
from scipy import spatial
import math

# reads a log file and returns a frequency distribution as a dict 从指定的CSV文件中读取用户的历史点击数据，输出三个
def load_click_distribution (menu, filename, normalize = True): # 从文件中加载点击历史记录
    history = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            history.append(row[0])
    return get_click_distribution(menu, history, normalize)
    

def get_click_distribution(menu, history, normalize = True):  # 获取点击分布 返回值三个 item点击频率 总点击数 点击历史记录
    frequency = {}  # 空字典frequency，用于存储每个菜单项的点击频率
    separator = "----"
    for command in menu:
        if command != separator:
            frequency[command] = 0  # 初始化菜单项的点击频率为0

    item_list = list(filter((separator).__ne__, menu)) #menu without separators filter()函数来过滤菜单列表menu，将其中不等于separator的元素提取出来，并转换为列表形式
    indexed_history = []
    for item in history:    #  计算menu的item的frequency
        indexed_history.append([item, item_list.index(item)])  # indexed_history [[potato,4],[gloves,0]...]
        if item not in list(frequency.keys()):
            frequency[item] = 1.
        else:
            frequency[item] += 1.
    if normalize:
        total_clicks = sum(list(frequency.values()))  # 总点击数
        for command in list(frequency.keys()):
            frequency[command] = round(frequency[command] / total_clicks,3)  # 保留三位小数
    return frequency, total_clicks, indexed_history  # 返回 item点击频率 总点击数 点击历史记录

# returns frequency distribution given a menu and history
def get_frequencies(menu, history, normalize = True):  # 获取菜单item的字典
    frequency = {}
    total_clicks = len(history)
    menu_items = list(filter(("----").__ne__, menu))
    for command in menu_items:
            frequency[command] = 0
            
    for row in history:
        if row[0] not in list(frequency.keys()):
            frequency[row[0]] = 1.
        else: 
            frequency[row[0]] += 1. 
    
    if normalize:
        for command in list(frequency.keys()):
            frequency[command] = round(frequency[command]/total_clicks, 3)
    
    return frequency, total_clicks

# Computes associatons based on word-embedding models. For each menu item, a list of associated items is returned
class KeyedVectors:
    pass


def compute_associations(menu, ft=None):  # 计算菜单项之间的关联度 词向量
    # Load pre-trained FT model from wiki corpus 加载预训练的 Word2Vec 模型 model，该模型用于计算词向量相似度
    # ft = fasttext.load_model('../fastText/models/cc.en.300.bin')
    # fasttext.util.reduce_model(ft, 100) 
    # Load pre-trained word2vec models. SO_vectors_200 = software engineering domain
    # model = KeyedVectors.load_word2vec_format('../fastText/models/SO_vectors_200.bin', binary=True)
    model = KeyedVectors.load_word2vec_format('../fastText/models/GoogleNews-vectors-negative300.bin', binary=True)  
    separator = "----"
    associations = {}  #关联字典
    associations_w2v = {}  # 基于 Word2Vec 的关联字典
    for command in menu:
        if command != separator:
            associations[command] = {command:1.0}
            associations_w2v[command] = {command:1.0}

    for i in menu:
        if i == separator: continue
        #Load word vector
        vector1 = ft.get_word_vector(i)
        vector1_word2vec = model.wv[i]
        for j in menu:
            if i == j or j == separator: continue
            vector2 = ft.get_word_vector(j)
            vector2_word2vec = model.wv[j]
            #Compute similarity score
            score = 1 - spatial.distance.cosine(vector1, vector2)
            score_word2vec = 1 - spatial.distance.cosine(vector1_word2vec, vector2_word2vec)
            print(i + "," + j + ": ft = " + str(round(score,3)) + " w2v = " + str(round(score_word2vec,3)) )
            associations[i][j] = score
            associations_w2v[i][j] = score_word2vec
        # 提前加载 FastText 模型 ft 或通过注释掉的代码进行预训练模型的加载和降维
    
    # print (associations)
    return associations

    # >>> vector1 = ft.get_word_vector('print')
    # >>> vector2 = ft.get_word_vector('duplicate')

    # >>> 1 - spatial.distance.cosine(vector1,vector2)
    # >>> 1 - spatial.distance.cosine(ft.get_word_vector('asparagus'),ft.get_word_vector('aubergine'))

def load_activations (history): # 激活值是根据用户的点击历史和时间间隔计算得出的 跟state.py里的一样 返回值是嵌套字典
    total_clicks = len(history)
    activations = {} # Activation per target per location
    duration_between_clicks = 20.0 # Wait time between two clicks
    session_interval = 50.0 # Wait time between 2 sessions
    session_click_length = 40 # Clicks per session
    total_sessions = math.ceil(total_clicks/session_click_length) # Number of sessions so far    
    for i in range(0, int(total_clicks)):
        session = math.ceil((i+1)/session_click_length) # Session index
        item = history[i][0]
        position = history[i][1]
        if item not in activations.keys(): activations[item] = {position:0} # Item has not been seen yet. Add to dictionary
        if position not in activations[item].keys(): activations[item][position] = 0 # Item not seen at this position yet. Add to item's dictionary
        time_difference = duration_between_clicks*(total_clicks - i) + (total_sessions - session)*session_interval # Difference between time now and time of click
        activations[item][position] += pow(time_difference, -0.5)
    return activations

def load_associations (menu, filename): #关联字典 key是item名字 value是与之关联的item列表,如果没有与之关联的 列表里就是item自身
    separator = "----"
    associations = {}
    for command in menu:
        if command != separator:
            associations[command] = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, skipinitialspace=True)
        for row in csv_reader:
            for item in row:
                if item in associations.keys():
                    associations[item] = associations[item] + row[0:]

    for key in associations:
        if associations[key] == []:
            associations[key] = [key]
    


    # with open(filename) as csv_file:
    #     csv_reader = csv.reader(csv_file)
    #     for row in csv_reader:
    #         if row[0] not in list(associations.keys()):
    #             associations[row[0]] = []
    #         associations[row[0]]= row[1:]
    return associations

def save_menu (menu, filename): # 保存menu到文件中
    f = open(filename, "w")
    for command in menu:
        f.write(command + "\n")
    f.close()

def load_menu (filename): # 导入 menu里的item
    menu = []
    f = open(filename, "r")
    for line in f:
        line = line.rstrip()
        if len(line) < 2: continue
        menu.append(line)
    return menu

# Returns association matrix for a menu using the associations dictionary
def get_association_matrix(menu, associations): # 根据菜单和关联字典生成关联矩阵
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
def get_sorted_frequencies(menu,frequency):  # 返回点击频率列表
    separator = "----"
    sorted_frequencies = []
    for k in range (0, len(menu)):
        if menu[k] == separator:
            sorted_frequencies.append(0.0)
        else:
            sorted_frequencies.append(frequency[menu[k]])
    return sorted_frequencies

    
def get_assoc_and_freq_list(state):
    separator = "----"
    associations = state.menu_state.associations
    frequency = state.user_state.freqdist
    menu = state.menu_state.menu
    # total_clicks = state.user_state.total_clicks
    # associations = load_associations(menu, filename)
    # frequency, total_clicks = load_click_distribution(menu, filename)
    assoc_list = []
    freq_list = []

    for k in range(0, len(menu)):
        if menu[k] in associations:
            for l in range(0, len(menu)):
                if menu[l] in associations[menu[k]]:
                    assoc_list.append(1.0)
                else:
                    assoc_list.append(0.0)
        else:
            for l in range (0, len(menu)):
                assoc_list.append(0.0)

    for k in range(0, len(menu)):
        if menu[k] == separator:
            freq_list.append(0.0)
        else:
            freq_list.append(frequency[menu[k]])
    return assoc_list, freq_list

def get_header_indexes(menu): # 返回菜单中所有组标题（header）的索引列表
        header_indexes = []
        separator = "----"
        groupboundary = False
        for i in range(0, len(menu)):
            if i == 0 or menu[i] == separator:
                groupboundary = True # Found a group start indicator
            if groupboundary and menu[i] != separator:
                header_indexes += [i] # First item of group (header)
                groupboundary = False
        return header_indexes