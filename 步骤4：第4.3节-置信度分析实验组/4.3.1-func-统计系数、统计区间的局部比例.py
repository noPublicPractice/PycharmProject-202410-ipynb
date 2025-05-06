import joblib
import numpy as np
from itertools import groupby
import json

class const:
    label_ch_list = ['房颤', '动脉粥样硬化', '心梗', '脑梗']
    label_en_list = ['AF', 'AS', 'MI', 'CI']
    folds = 6
    great_random_state = [15000, 13000, 12000, 15000]
    model_name = 'LGBM'
    model_values_joblib_name = r'..\步骤2：第3章与第4.1节-训练模型、生成风险概率矩阵\模型数据pkl\%s模型pkl-Fold=%d-random_state=%d-exclude_Fold=%d-model_name=%s.pkl'
    au_roc = {"AF": [0.881674, 0.881028, 0.881605, 0.882830, 0.881821, 0.881481], "AS": [0.867298, 0.866973, 0.863639, 0.865253, 0.868168, 0.863537],
              "MI": [0.920794, 0.920173, 0.920058, 0.920949, 0.919216, 0.919803], "CI": [0.842362, 0.839910, 0.839551, 0.843969, 0.839547, 0.839374]}
    density = 200  # 密度
    interval_num = 20  # groupby()统计区间数量
    inner_interval_num = int(density / interval_num)  # 每个groupby()统计区间，内部的小区间数量
    do_write_json_path = r'..\绘图数据【保留】\%s%s-threshold-with-statistical-coefficients-dict-interval_num=%d.json'
def test_data_label_index(test_data_label):
    label_list = np.array(test_data_label)
    zeros_list = np.zeros(len(test_data_label))
    positive_index = np.where(label_list != zeros_list)[0]
    negative_index = np.where(label_list == zeros_list)[0]
    return positive_index, negative_index
def calc_pred_pro_weighted_sum(label_en, pred_pro_list):  # 计算风险概率基于auroc加权的求和值
    pred_pro_k = np.array(const.au_roc[label_en]) / sum(const.au_roc[label_en])
    pred_pro_arr = np.array(pred_pro_list)
    pred_pro_weighted_sum = np.dot(pred_pro_k, pred_pro_arr)  # 风险概率
    return pred_pro_weighted_sum
def do_group(pre):
    group_dict = {}
    for k, g in groupby(sorted(pre), key=lambda x: int(x * const.interval_num)):
        group_name = '%.3f-%.3f' % (k / const.interval_num, (k + 1) / const.interval_num)
        group_dict[group_name] = len(list(g))
    # 笔记：将列表按数值区间计元素个数/列表按区间分组统计各组个数
    # 笔记：其中g是迭代器，只能被遍历一次（即在同一次运行的同一次循环中，list(g)只能使用一次）
    # 试验：取 interval_num=20时，0.3与0.349999999都被分在"0.3-0.35"这一组中
    return group_dict
def update_statistical_coefficients_2(partial_pro_dict):
    inner_interval_statistical_coefficients = {}
    threshold_spacing = 1 / const.interval_num  # 区间端点的间隔
    inner_interval = 1 / const.density  # 小区间端点的间隔
    more_half_inner_interval_num = int(1.5 * const.inner_interval_num)
    for index, interval_endpoint in enumerate(np.linspace(threshold_spacing, 1.0 - threshold_spacing, const.interval_num - 1)):
        # 左侧groupby()统计区间的属性
        left_interval_left_endpoint = interval_endpoint - threshold_spacing  # 统计区间的左端点
        left_interval_middle = (left_interval_left_endpoint + interval_endpoint) / 2  # 统计区间的中点
        left_interval_partial_pro = partial_pro_dict['%.3f-%.3f' % (left_interval_left_endpoint, interval_endpoint)]  # 统计区间的局部比例
        # 右侧groupby()统计区间的属性
        right_interval_right_endpoint = interval_endpoint + threshold_spacing  # 统计区间的右端点
        right_interval_middle = (interval_endpoint + right_interval_right_endpoint) / 2  # 统计区间的中点
        right_interval_partial_pro = partial_pro_dict['%.3f-%.3f' % (interval_endpoint, right_interval_right_endpoint)]  # 统计区间的局部比例
        # 构造赋值列表
        if index == 0 or index == const.interval_num - 2:
            inner_interval_partial_pro = np.linspace(left_interval_partial_pro, right_interval_partial_pro, more_half_inner_interval_num)
        else:
            inner_interval_partial_pro = np.linspace(left_interval_partial_pro, right_interval_partial_pro, const.inner_interval_num)
        # 小区间“统计系数”赋值
        if index == 0:
            inner_interval_endpoint_list = enumerate(np.linspace(left_interval_left_endpoint + inner_interval, right_interval_middle, more_half_inner_interval_num))
        elif index == const.interval_num - 2:
            inner_interval_endpoint_list = enumerate(np.linspace(left_interval_middle + inner_interval, right_interval_right_endpoint, more_half_inner_interval_num))
        else:
            inner_interval_endpoint_list = enumerate(np.linspace(left_interval_middle + inner_interval, right_interval_middle, const.inner_interval_num))
        for inner_index, inner_interval_right_endpoint in inner_interval_endpoint_list:
            inner_group_name = '%.3f-%.3f' % (inner_interval_right_endpoint - inner_interval, inner_interval_right_endpoint)
            inner_interval_statistical_coefficients[inner_group_name] = inner_interval_partial_pro[inner_index]
    return inner_interval_statistical_coefficients
def update_statistical_coefficients(positive_group_dict, negative_group_dict):
    ## create saved dict
    statistical_coefficients = {"partial_positive_pro": {}, "partial_negative_pro": {}}
    inner_interval_statistical_coefficients = {"partial_positive_pro": {}, "partial_negative_pro": {}}
    ## update interval dict
    threshold_spacing = 1 / const.interval_num  # 区间端点的间隔--分类阈值间隔
    for index, interval_right_endpoint in enumerate(np.linspace(threshold_spacing, 1.0, const.interval_num)):
        interval_left_endpoint = interval_right_endpoint - threshold_spacing
        group_name = '%.3f-%.3f' % (interval_left_endpoint, interval_right_endpoint)
        partial_positive_num = positive_group_dict[group_name] if group_name in positive_group_dict else 0
        partial_negative_num = negative_group_dict[group_name] if group_name in negative_group_dict else 0
        partial_num = partial_positive_num + partial_negative_num
        if partial_num == 0:
            statistical_coefficients["partial_positive_pro"][group_name] = 1.0
            statistical_coefficients["partial_negative_pro"][group_name] = 1.0
        else:
            statistical_coefficients["partial_positive_pro"][group_name] = partial_positive_num / partial_num  # 正类统计系数--局部正类比例
            statistical_coefficients["partial_negative_pro"][group_name] = partial_negative_num / partial_num  # 负类统计系数--局部负类比例
    ## update inner interval dict
    inner_interval_statistical_coefficients["partial_positive_pro"] = update_statistical_coefficients_2(statistical_coefficients["partial_positive_pro"])
    inner_interval_statistical_coefficients["partial_negative_pro"] = update_statistical_coefficients_2(statistical_coefficients["partial_negative_pro"])
    return inner_interval_statistical_coefficients
def save_json(label_en, label_ch, statistical_coefficients):
    statistical_coefficients_dict = {
        "partial_positive_pro": statistical_coefficients["partial_positive_pro"],
        "partial_negative_pro": statistical_coefficients["partial_negative_pro"]
    }
    now_json_path_file_name = const.do_write_json_path % (label_en, label_ch, const.interval_num)
    fp = open(now_json_path_file_name, 'w', encoding='utf-8')
    json.dump(statistical_coefficients_dict, fp, indent=4, ensure_ascii=False, sort_keys=False)
    fp.close()
def do_work():
    for label_ch, label_en, random_state in zip(const.label_ch_list, const.label_en_list, const.great_random_state):
        now_model_values_joblib_name = const.model_values_joblib_name % (label_en, const.folds, random_state, 1, const.model_name)
        vectorizer, model, test_data_label, pred_pro, pred_lr = joblib.load(open(now_model_values_joblib_name, "rb"))
        positive_index, negative_index = test_data_label_index(test_data_label)  # 标记空间中的正类索引、标记空间中的负类索引
        ## 计算风险概率均值
        pred_pro_list = []  # 风险概率列表
        for fold in range(const.folds):
            now_model_values_joblib_name = const.model_values_joblib_name % (label_en, const.folds, random_state, fold + 1, const.model_name)
            vectorizer, model, test_data_label, pred_pro, pred_lr = joblib.load(open(now_model_values_joblib_name, "rb"))
            pred_pro_list.append(list(pred_pro))
        pred_pro_weighted_sum = calc_pred_pro_weighted_sum(label_en, pred_pro_list)
        ## 风险概率分类阈值，对应的准确率
        positive_group_dict = do_group(pred_pro_weighted_sum[positive_index])
        negative_group_dict = do_group(pred_pro_weighted_sum[negative_index])
        statistical_coefficients = update_statistical_coefficients(positive_group_dict, negative_group_dict)
        save_json(label_en, label_ch, statistical_coefficients)
if __name__ == "__main__":
    do_work()
