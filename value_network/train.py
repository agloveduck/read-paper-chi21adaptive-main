#!/usr/bin/env python3
# coding: utf-8

import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import tensorflow as tf


# Boost GPU usage.
conf = tf.compat.v1.ConfigProto()
conf.gpu_options.allow_growth = True
tf.compat.v1.Session(config=conf)

# Maximum number of items in a given menu, including separators.
MAX_MENU_ITEMS = 20  # 菜单中的最大项数，包括分隔符
# Size of the one-hot encoded vectors. This value should be large enough to avoid hashing collisions.
ENC_VOCAB_SIZE = 90  # one-hot 编码字典大小


def load_data(filepath):  # 从文件中加载训练数据 返回价值网络的 输入 输出
    X1, X2, X3, X4 = [], [], [], []
    y1, y2, y3 = [], [], []

    with open(filepath) as f:
        for line in f.read().splitlines():
            (serial, forage, recall), (target_menu, diff_freq, diff_asso), exposed = format_row(line)

            X1.append(target_menu)
            X2.append(diff_freq)
            X3.append(diff_asso)
            X4.append(exposed)

            y1.append(serial)
            y2.append(forage)
            y3.append(recall)

    return (np.array(X1), np.array(X2), np.array(X3), np.array(X4)), (np.array(y1), np.array(y2), np.array(y3))


def format_row(line):  # 将 pump.py 产生的result.txt 文件 按行解析返回价值网络的输入
    (serial, forage, recall), (source_menu, source_freq, source_asso), (target_menu, target_freq, target_asso), exposed = parse_row(line)

    adap_menu, diff_freq, diff_asso = parse_user_input(source_menu, source_freq, source_asso, target_menu, target_freq, target_asso)

    return (serial, forage, recall), (adap_menu, diff_freq, diff_asso), exposed


def parse_row(line):  #
    # Row format is "[serial,forage,recall][source_menu][source_freq][source_asso][target_menu][target_freq][target_asso][exposed]"
    tokens = line[1:-1].split('][')
    n_toks = len(tokens)
    assert n_toks == 8, 'There are {} tokens, but I expected 8'.format(n_toks)  #根据预期的格式，应该有 8 个字段。如果实际的字段数不是 8，则会引发断言错误

    # FIXME: We should agree on a parser-friendly row format.
    serial, forage, recall = list(map(float, tokens[0].split(', ')))

    source_menu = list(map(lambda x: x.replace("'", ''), tokens[1].split(', ')))
    source_freq = list(map(float, tokens[2].split(', ')))
    source_asso = list(map(float, tokens[3].split(', ')))

    target_menu = list(map(lambda x: x.replace("'", ''), tokens[4].split(', ')))
    target_freq = list(map(float, tokens[5].split(', ')))
    target_asso = list(map(float, tokens[6].split(', ')))

    # Currently there's only one extra feat, but wrap it as list in case we add more feats in the future.
    exposed = [bool(tokens[7])]

    return (serial, forage, recall), (source_menu, source_freq, source_asso), (target_menu, target_freq, target_asso), exposed


def parse_user_input(source_menu, source_freq, source_asso, target_menu, target_freq, target_asso):  # 返回 adap_menu, adj(diff_freq), diff_asso
    # Encode adapted menu as integers and compute the difference between previous and current menu configuration.
    adap_menu = onehot_menu(target_menu)  # onehot编码的adap_menu
    # Adjust remaining menu items with zeros (reserved value) at the end.
    adap_menu = adj(adap_menu, value=[0])  # 整形，补0

#    # Experimental: ignore differences w.r.t source menu.
#    num_cols = len(target_freq)
#    tgt_asso = np.array(target_asso).reshape((num_cols, num_cols))
#    tgt_asso = adj([adj(item) for item in tgt_asso], [0]*MAX_MENU_ITEMS)
#    tgt_asso = tgt_asso.reshape((MAX_MENU_ITEMS*MAX_MENU_ITEMS,))
#    return adap_menu, adj(target_freq), tgt_asso

    # Ensure that all vectors have the same length.
    max_freq_len = max(len(source_freq), len(target_freq))
    max_asso_len = max(len(source_asso), len(target_asso))
    source_freq = pad(source_freq, max_freq_len)
    target_freq = pad(target_freq, max_freq_len)
    source_asso = pad(source_asso, max_asso_len)
    target_asso = pad(target_asso, max_asso_len)

    diff_freq = np.diff([source_freq, target_freq], axis=0).flatten()  # 展平成1维
    diff_asso = np.diff([source_asso, target_asso], axis=0).flatten()

    # Ensure there is a change in freq distribution, otherwise `diff_freq` would be always zero.
    if np.array_equal(source_freq, target_freq):
        diff_freq = source_freq

    # The association matrix list is given as a flat vector, so reshape it before padding.
    # Notice that we read the number of items BEFORE padding `diff_freq`.
    num_rows = len(diff_freq)  # num_rows 的值等于源菜单频率向量和目标菜单频率向量中较长的那个向量
    num_cols = len(diff_asso)//num_rows
    diff_asso = diff_asso.reshape((num_cols, num_rows))  # 整形
    diff_asso = adj([adj(item) for item in diff_asso], [0]*MAX_MENU_ITEMS)
    diff_asso = diff_asso.reshape((MAX_MENU_ITEMS*MAX_MENU_ITEMS,))

    return adap_menu, adj(diff_freq), diff_asso


def pad(l, size, value=0):  # 向量补0
    return l + [value] * abs((len(l)-size))  #  计算向量长度


def adj(vec, value=0):  # 调整向量大小 使其具有固定的长度 MAX_MENU_ITEMS
    N = len(vec)
    d = MAX_MENU_ITEMS - N
    if d < 0:
        # Truncate vector.
        vec = vec[:MAX_MENU_ITEMS]  # 向量长度超过目标长度，将向量进行截断，只保留前 MAX_MENU_ITEMS 个元素
    elif d > 0:
        # Pad vector with zeros (reserved value) at the *end* of the vector.
        vec = list(vec) + [value for _ in range(d)]  #  向量长度不足目标长度，将向量末尾填充 d 个 0
    return np.array(vec)


def onehot_menu(items):  # 将菜单进行 one-hot 编码
    # FIXME: We should agree on a single-word menu separator, because '----' is conflicting with the built-in text parser.
    enc_menu = [tf.keras.preprocessing.text.one_hot(w, ENC_VOCAB_SIZE, filters='') for w in items]
    return enc_menu


def create_model(adap_menu, diff_freq, diff_asso, xtra_feat):
    # The provided sample args are needed to get the input shapes right.
    # For example, the network capacity is bounded by the (max) number of menu items.
    num_items = diff_freq.shape[0]

    def menu_head(inputs):  # 输入 Adapted menu 输出 在文中图6 concatenator层的输入
        m = tf.keras.layers.Embedding(ENC_VOCAB_SIZE, num_items, input_length=num_items)(inputs)
        m = tf.keras.layers.Flatten()(m)
        m = tf.keras.layers.Dropout(0.5)(m)
        m = tf.keras.layers.Dense(num_items//2)(m)
        m = tf.keras.Model(inputs=inputs, outputs=m)
        return m

    def freq_head(inputs):  # 输入点击频率差分矩阵 输出 在文中图6 concatenator层的输入
        f = tf.keras.layers.Reshape((num_items, 1))(inputs)
        f = tf.keras.layers.LSTM(num_items, activation='relu')(f)
        f = tf.keras.layers.Dropout(0.5)(f)
        f = tf.keras.layers.Dense(num_items//2)(f)
        f = tf.keras.Model(inputs=inputs, outputs=f)
        return f

    def asso_head(inputs):  # 输入菜单关联性差分矩阵 输出 在文中图6 concatenator层的输入
        a = tf.keras.layers.Reshape((num_items, num_items))(inputs)
        a = tf.keras.layers.LSTM(num_items*2, activation='relu')(a)
        a = tf.keras.layers.Dropout(0.5)(a)
        a = tf.keras.layers.Dense(num_items//2)(a)
        a = tf.keras.Model(inputs=inputs, outputs=a)
        return a

    def serial_tail(inputs):  # 输入 文中图6 concatenator层的输出部分 后续网络
        s = tf.keras.layers.Dense(num_items//2)(inputs)
        s = tf.keras.layers.Dropout(0.5)(s)
        s = tf.keras.layers.Dense(1)(s)
        s = tf.keras.layers.Activation('linear', name='serial_output')(s)
        return s

    def forage_tail(inputs):
        f = tf.keras.layers.Dense(num_items//2)(inputs)
        f = tf.keras.layers.Dropout(0.5)(f)
        f = tf.keras.layers.Dense(1)(f)
        f = tf.keras.layers.Activation('linear', name='forage_output')(f)
        return f

    def recall_tail(inputs):
        r = tf.keras.layers.Dense(num_items//2)(inputs)
        r = tf.keras.layers.Dropout(0.5)(r)
        r = tf.keras.layers.Dense(1)(r)
        r = tf.keras.layers.Activation('linear', name='recall_output')(r)
        return r

    input_menu = tf.keras.layers.Input(shape=adap_menu.shape, name='menu')
    input_freq = tf.keras.layers.Input(shape=diff_freq.shape, name='priors')
    input_asso = tf.keras.layers.Input(shape=diff_asso.shape, name='associations')
    input_feat = tf.keras.layers.Input(shape=xtra_feat.shape, name='features')  # 暴不暴露给用户

    menu = menu_head(input_menu)
    freq = freq_head(input_freq)
    asso = asso_head(input_asso)

    combined_head = tf.keras.layers.concatenate([menu.output, freq.output, asso.output, input_feat])
    serial = serial_tail(combined_head)
    forage = forage_tail(combined_head)
    recall = recall_tail(combined_head)

    # Hereby I compose the almighty value network model.
    model = tf.keras.Model(inputs=[menu.input, freq.input, asso.input, input_feat], outputs=[serial, forage, recall])
    losses = {'serial_output': 'mse', 'forage_output': 'mse', 'recall_output': 'mse'}
    model.compile(optimizer='rmsprop', loss=losses, metrics=['mse', 'mae'])

    return model



if __name__ == '__main__':
    # Input can be either a list of files or a directory.
    train_inputs = sys.argv[1:]  # 确定了训练数据的输入路径
    #  sys.argv[1:] output/results_vn_214343.txt
    # Collect all training files first.
    tr_files = []  # 创建一个空列表用于收集所有训练文件的路径

    # Input can be either a list of files or a directory.
    train_inputs = sys.argv[1:]  # 确定了训练数据的输入路径

    # Collect all training files first.
    tr_files = []  # 创建一个空列表用于收集所有训练文件的路径

    for tr_input in train_inputs:  # 循环遍历输入的训练数据文件或目录
        tr_input_absolute = os.path.abspath(tr_input)  # Convert the relative path to absolute path
        if os.path.isdir(tr_input_absolute):  # 判断输入是否为目录
            # If the input is a directory, recursively walk through the directory to find all files with the '.txt' extension.
            for path, directories, files in os.walk(tr_input_absolute):
                for f in files:
                    if f.endswith('.txt'):  # 判断文件是否以'.txt'结尾
                        file_path = os.path.join(path, f)  # 获取文件的绝对路径
                        tr_files.append(file_path)  # 将文件的绝对路径添加到 tr_files 列表中

        elif os.path.isfile(tr_input_absolute):  # 判断输入是否为文件
            # If the input is a file, add its path directly to the tr_files list.
            tr_files.append(tr_input_absolute)  # 将文件的绝对路径添加到 tr_files 列表中

    print(tr_files)
    X1, X2, X3, X4 = [], [], [], []
    y1, y2, y3 = [], [], []

    for f in tr_files:
        (X1_, X2_, X3_, X4_), (y1_, y2_, y3_) = load_data(f)
        X1 = np.concatenate((X1, X1_)) if len(X1) > 0 else X1_
        X2 = np.concatenate((X2, X2_)) if len(X2) > 0 else X2_
        X3 = np.concatenate((X3, X3_)) if len(X3) > 0 else X3_
        X4 = np.concatenate((X4, X4_)) if len(X4) > 0 else X4_
        y1 = np.concatenate((y1, y1_)) if len(y1) > 0 else y1_
        y2 = np.concatenate((y2, y2_)) if len(y2) > 0 else y2_
        y3 = np.concatenate((y3, y3_)) if len(y3) > 0 else y3_

    # Provide one sample of the input data to the model.+
    3

    model = create_model(X1[0], X2[0], X3[0], X4[0])

#    model.summary()  绘制模型结构图
#    tf.keras.utils.plot_model(model, show_shapes=False, to_file='value_network.png')
#    tf.keras.utils.plot_model(model, show_shapes=True, to_file='value_network_with_shapes.png')
#    tf.keras.utils.plot_model(model, show_shapes=False, show_layer_names=False, to_file='value_network_blocks.png')

    from time import time
    now = int(time())
    #  模型训练中使用的回调函数
    # 训练过程中的日志信息写入到 TensorBoard 日志目录中 在20.轮数内指标没有改善，则提前终止训练 恢复到在验证集上表现最好的模型参数
    cbs = [
        tf.keras.callbacks.TensorBoard(log_dir='./training_logs_{}'.format(now)),
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
    ]
    #  训练数据的 20% 作为验证集
    model.fit([X1, X2, X3, X4], [y1, y2, y3], validation_split=0.2, epochs=200, batch_size=32, callbacks=cbs)
    model.save('value_network.h5')
