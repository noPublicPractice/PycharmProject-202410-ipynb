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
    density = 200
    do_write_json_path = r'..\绘图数据【保留】\%s%s-threshold-with-acc-ppv-tpr-dict-density=%d.json'
def test_data_label_index(test_data_label):
    # if type(test_data_label) is np.ndarray:
    #     test_data_label=list(test_data_label)
    label_list = np.array(test_data_label)
    zeros_list = np.zeros(len(test_data_label))
    positive_index = np.where(label_list != zeros_list)[0]
    negative_index = np.where(label_list == zeros_list)[0]
    return positive_index, negative_index
def calc_pred_pro_weighted_sum(label_en, pred_pro_list):  # 计算风险概率基于auroc加权的求和值
    pred_pro_k = np.array(const.au_roc[label_en]) / sum(const.au_roc[label_en])
    pred_pro_arr = np.array(pred_pro_list)
    pred_pro_weighted_sum = np.dot(pred_pro_k, pred_pro_arr)  # 风险概率
    # pred_pro_mean = np.mean(pred_pro_arr.T, axis=1)
    return pred_pro_weighted_sum
def do_group(pre):
    group_dict = {}
    for k, g in groupby(sorted(pre), key=lambda x: int(x * const.density)):
        group_name = '%.3f-%.3f' % (k / const.density, (k + 1) / const.density)
        group_dict[group_name] = len(list(g))
    # 笔记：将列表按数值区间计元素个数/列表按区间分组统计各组个数
    # 笔记：其中g是迭代器，只能被遍历一次（即在同一次运行的同一次循环中，list(g)只能使用一次）
    # 试验：取 density=20时，0.3与0.349999999都被分在"0.3-0.35"这一组中
    return group_dict
def calc_accuracy(tp, tn, amount_sum):
    return (tp + tn) / amount_sum
def calc_ppv(tp, fp):
    if tp + fp == 0:
        return 1.0
    return tp / (tp + fp)
def calc_tpr(tp, fn):
    return tp / (tp + fn)
def calc_f1(tp, fn, fp):
    return 2 * tp / (2 * tp + fn + fp)  # 2*ppv*tpr/(ppv+tpr)
def calc_threshold_with_assessed_values(positive_group_dict, negative_group_dict, positive_n, negative_n):
    tp = positive_n  # 正类，被预测成正类的个数
    fn = 0  # 正类，被预测成负类的个数
    fp = negative_n  # 负类，被预测成正类的个数
    tn = 0  # 负类，被预测成负类的个数
    amount_sum = tp + fn + fp + tn  # 总数
    ## create saved list
    threshold_with_assessed_values = {
        "classify_threshold": np.zeros(const.density + 1),
        "acc":                np.zeros(const.density + 1),
        "ppv":                np.zeros(const.density + 1),
        "tpr":                np.zeros(const.density + 1),
        "f1":                 np.zeros(const.density + 1)
    }
    ## update saved list
    threshold_interval = 1 / const.density  # 分类阈值的间隔
    for index, classify_threshold in enumerate(np.linspace(0.0, 1.0, const.density + 1)):
        if index:  # 当分类阈值为0时，跳过此步骤
            group_name = '%.3f-%.3f' % (classify_threshold - threshold_interval, classify_threshold)
            if group_name in positive_group_dict:
                # print('positive %d' % positive_group_dict[group_name])
                tp = tp - positive_group_dict[group_name]
                fn = fn + positive_group_dict[group_name]
            if group_name in negative_group_dict:
                # print('negative %d' % negative_group_dict[group_name])
                fp = fp - negative_group_dict[group_name]
                tn = tn + negative_group_dict[group_name]
        threshold_with_assessed_values["classify_threshold"][index] = round(classify_threshold, 3)
        threshold_with_assessed_values["acc"][index] = calc_accuracy(tp, tn, amount_sum)
        threshold_with_assessed_values["ppv"][index] = calc_ppv(tp, fp)
        threshold_with_assessed_values["tpr"][index] = calc_tpr(tp, fn)
        threshold_with_assessed_values["f1"][index] = calc_f1(tp, fn, fp)
    return threshold_with_assessed_values
def save_json(label_en, label_ch, threshold_with_assessed_values):
    # for i in range(np.size(threshold_with_assessed_values, 1)):
    #     print('分类阈值=%.3f，accuracy=%.3f' % (threshold_with_assessed_values[0][i], threshold_with_assessed_values[1][i]))  # 准确率
    threshold_with_assessed_values_dict = {
        "classify_threshold": list(threshold_with_assessed_values["classify_threshold"]),
        "acc":                list(threshold_with_assessed_values["acc"]),
        "ppv":                list(threshold_with_assessed_values["ppv"]),
        "tpr":                list(threshold_with_assessed_values["tpr"]),
        "acc_multiply_tpr":   list(threshold_with_assessed_values["acc"] * threshold_with_assessed_values["tpr"]),
        "f1":                 list(threshold_with_assessed_values["f1"])
    }
    now_json_path_file_name = const.do_write_json_path % (label_en, label_ch, const.density)
    fp = open(now_json_path_file_name, 'w', encoding='utf-8')
    json.dump(threshold_with_assessed_values_dict, fp, indent=4, ensure_ascii=False, sort_keys=False)
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
        threshold_with_assessed_values = calc_threshold_with_assessed_values(positive_group_dict, negative_group_dict, len(positive_index), len(negative_index))
        save_json(label_en, label_ch, threshold_with_assessed_values)
if __name__ == "__main__":
    do_work()
