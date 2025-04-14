import pandas as pd
import numpy as np
from sklearn.utils import shuffle

class const:
    label = 'CI'  # AF,AS,MI,CI
    label_ch = '脑梗'  # 房颤,动脉粥样硬化,心梗,脑梗
    folds = 6
    great_random_state = 15000
    whole_training_csv_path = r'A疾病源数据集csv\df_training_%s.csv' % label
    folds_training_csv_path = r'交叉训练集csv、pickle\%s交叉训练集-Fold=%d-random_state=%d-exclude_Fold=%d.csv'
    folds_validate_csv_path = r'交叉训练集csv、pickle\%s交叉验证集-Fold=%d-random_state=%d-Fold=%d.csv'
    whole_testing_csv_path = r'A疾病源数据集csv\df_eval_common.csv'
    save_testing_csv_path = r'交叉训练集csv、pickle\公共测试集.csv'
def create_training_csv():  # 生成交叉训练集及交叉验证集
    ## 读取文件
    whole_training_data = pd.read_csv(const.whole_training_csv_path, sep=';')  # 读取文件
    whole_training_data = whole_training_data[~(whole_training_data['入院年份'] == '2008 - 2010')]
    whole_training_data.reset_index(drop=True, inplace=True)
    # 笔记：数据源文件已经使用'const.label_ch'列，区分房颤患者、非房颤患者
    # 笔记：排除2010年之前数据，降低数据迁移影响。
    # 笔记：reset_index用于重置数据帧的索引，用'drop=True'来避免将旧索引添加为列
    
    ## 患者ID去重，打乱患者ID，用分割点拆分患者ID，优化分层采样效果
    patient_id_unique = whole_training_data['患者ID'].unique()
    split_point = np.linspace(0, len(patient_id_unique), const.folds + 1, dtype='int')  # 分割点
    whole_training_positive_rate = sum(whole_training_data[const.label_ch]) / len(whole_training_data)
    folds_min_mean_positive_var = 0.5
    # 笔记：whole_training_data['患者ID']的数据格式是Series，Series.unique()返回一个包含Series或DataFrame中唯一值的数组（对'患者ID'列去重）
    # 笔记：分割点依次为16.7%、33.3%、50.0%、66.7%、83.3%，整个训练集均分为六个Fold，每次用一个Fold作为验证集，其余合成一个训练集
    # 笔记：希望让各个Fold的房颤阳性数据项比例尽可能接近整个训练集的房颤阳性数据项比例：16.43%（阳性总数：25254；阴性总数：128416）
    # 笔记：优化目标是使"folds_min_mean_positive_var"取到尽可能小的数值，使得分层采样的效果尽可能好
    
    for random_state_num in range(10000, 20000, 1000):  # 遍历一些随机数
        patient_id_after_shuffle = shuffle(patient_id_unique, random_state=random_state_num)
        patient_id_split_folds = [patient_id_after_shuffle[split_point[i]:split_point[i + 1]] for i in range(const.folds)]
        folds_training_data = [whole_training_data[whole_training_data['患者ID'].isin(patient_id_split_folds[i])] for i in range(const.folds)]
        folds_positive_var = [abs(sum(folds_training_data[i][const.label_ch]) / len(folds_training_data[i]) - whole_training_positive_rate) ** 2 for i in range(const.folds)]
        folds_mean_positive_var = sum(folds_positive_var) / const.folds
        if folds_mean_positive_var < folds_min_mean_positive_var:
            const.great_random_state = random_state_num
            folds_min_mean_positive_var = folds_mean_positive_var
            print(const.great_random_state, folds_min_mean_positive_var)
            great_folds_training_data = folds_training_data
    # 笔记：sklearn.utils.shuffle()用于打乱顺序，加一个固定的random_state参数后，每次运行shuffle函数得到打乱后的结果都是相同的
    # 笔记：构造顺序：打乱后的"病人ID" -> "病人ID"分割成Folds -> 构造各Fold -> 计算各Fold的var值
    # 笔记："sum(df_training_fold[i][const.label_ch]) / len(df_training_fold[i])"表示：单个Fold中，房颤阳性数据项比例
    
    ## 区分训练集fold和验证集fold，合并训练集fold，打乱顺序，保存文件
    for i in range(const.folds):
        folds_training_index = list(range(const.folds))
        folds_validate_index = folds_training_index.pop(i)
        # folds_training = pd.concat([great_folds_training_data[folds_training_index]])  # 合并
        temp = pd.concat([great_folds_training_data[folds_training_index[0]], great_folds_training_data[folds_training_index[1]]])
        for j in range(2, const.folds-1):
            temp = pd.concat([temp, great_folds_training_data[folds_training_index[j]]])
        folds_training = temp
        folds_training = shuffle(folds_training, random_state=const.great_random_state)  # 打乱顺序
        folds_training.to_csv(const.folds_training_csv_path % (const.label, const.folds, const.great_random_state, i + 1), sep=";", index=False)  # 保存训练集
        folds_validate = great_folds_training_data[folds_validate_index]
        folds_validate.to_csv(const.folds_validate_csv_path % (const.label, const.folds, const.great_random_state, i + 1), sep=";", index=False)  # 保存验证集
def create_testing_csv():  # 生成测试集
    # Testing data 始终不变
    whole_testing_data = pd.read_csv(const.whole_testing_csv_path, sep=';')  # 读取文件
    whole_testing_data = whole_testing_data[~(whole_testing_data['入院年份'] == '2008 - 2010')]
    whole_testing_data.reset_index(drop=True, inplace=True)
    whole_testing_data.to_csv(const.save_testing_csv_path, sep=";", index=False)  # 保存文件
if __name__ == '__main__':
    create_training_csv()
    # create_testing_csv()
