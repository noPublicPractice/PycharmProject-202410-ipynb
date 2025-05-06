import os
import pickle
import pandas as pd
import numpy as np

class const:
    label = 'CI'  # AF,AS,MI,CI
    label_ch = '脑梗'  # 房颤,动脉粥样硬化,心梗,脑梗
    folds = 6
    great_random_state = 15000
    save_testing_csv_path = r'交叉训练集csv、pickle\公共测试集.csv'
    folds_training_csv_pa = r'交叉训练集csv、pickle\%s交叉训练集-Fold=%d-random_state=%d-exclude_Fold=%d.csv'
    data_frame_pickle_pat = r'交叉训练集csv、pickle\%s样本空间、标记空间-Fold=%d-random_state=%d-exclude_Fold=%d.pickle'
    features_training = ['性别', '年龄组', '入院类型', '诊断史', '手术史', '实验室结果', '入院地点', '保险', '语言', '婚姻状况', '种族']  # 定义训练用特征
def csv_to_data_frame(file_name):
    df_data = pd.read_csv(file_name, sep=';')  # 导入数据集
    df_data[const.features_training] = df_data[const.features_training].astype(str)  # 确保所有特征值都是字符串
    ## 每一个数据项的各个属性都拼接成字符串
    split_ch = np.full(len(df_data), ',')  # Numpy初始化全是","的ndarray数组，个数为df_data的行数
    str_data = df_data[const.features_training[0]]  # Series数据结构：取值，df_data['性别']
    for i in range(1, len(const.features_training)):
        temp_lst = np.char.add(split_ch, df_data[const.features_training[i]].to_numpy())  # NumPy中两个字符串数组的元素串联
        temp_lst[temp_lst==',nan'] = ''  # numpy数组中值的替换
        str_data = np.char.add(str_data, temp_lst)
    # 笔记：df_data[const.features_training[i]]的数据格式是Series，Series.to_numpy()表示将Series转换为Numpy数组
    # 笔记：对于所有元素都是字符串的二维矩阵，np.char.add()将每一行字符串拼成一个字符串
    # 笔记：对于temp_lst，必须要保证，只有这种情况：',nan.{0},'；没有这种情况：',nan.{n},'其中n>=1
    ## 构建数据框
    data_frame = pd.DataFrame({'住宿ID': df_data['住宿ID'], '观察值': str_data.to_list(), const.label_ch: df_data[const.label_ch]})
    return data_frame
def create_pickle_frame():
    test_data_frame = csv_to_data_frame(const.save_testing_csv_path)
    for i in range(const.folds):
        pickle_path = const.data_frame_pickle_pat % (const.label, const.folds, const.great_random_state, i + 1)
        if not os.path.exists(pickle_path):  # 如果pickle文件未创建
            train_data_frame_list = csv_to_data_frame(const.folds_training_csv_pa % (const.label, const.folds, const.great_random_state, i + 1))
            pickle.dump((train_data_frame_list, test_data_frame), open(pickle_path, "wb"))  # 保存变量到文件.注意:二进制模式不采用编码参数
if __name__=="__main__":
    create_pickle_frame()
