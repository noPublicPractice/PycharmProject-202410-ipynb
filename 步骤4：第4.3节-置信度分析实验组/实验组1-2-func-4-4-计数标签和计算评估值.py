import json
import numpy as np
from sklearn.metrics import roc_curve, auc

class const:
    label_ch_list = ['房颤', '动脉粥样硬化', '心梗', '脑梗']
    label_en_list = ['AF', 'AS', 'MI', 'CI']
    popt = {'AF': [-0.12624826, 0.20331957, 0.43457583], 'AS': [-0.53775715, 1.87577197, -0.67785164],
            'MI': [-0.51259901, 1.71942337, -0.54627487], 'CI': [-1.35737578, 3.64757585, -1.64243702]}
    confidence_score_f = 'std×pp'
    do_write_predict_information_json_path = r'..\绘图数据【保留】\%s%s-predict-information-dict-func=%s.json'
def prepare_predict_information(label_en, label_ch):
    now_json_path_file_name = const.do_write_predict_information_json_path % (label_en, label_ch, const.confidence_score_f)
    predict_information_dict = json.load(open(now_json_path_file_name, 'r', encoding='utf-8'))
    pred_pro_weighted_sum = np.array(predict_information_dict["pred_pro_weighted_sum"])
    pred_pro_confidence_score = np.array(predict_information_dict["pred_pro_confidence_score"])
    test_data_label_arr = np.array(predict_information_dict["test_data_label_arr"]).astype(int)  # 问题：JSON不能保存np.int，所以用float代替；此处转化回来
    pred_label_arr = np.array(predict_information_dict["pred_label_arr"]).astype(int)
    return pred_pro_weighted_sum, pred_pro_confidence_score, test_data_label_arr, pred_label_arr
def fit_func(x, a, b, c):  # 变量一定要放在第一个位置
    return a * x ** 2 + b * x + c
def prepare_predict_information_dict(label_en, pred_pro_weighted_sum, pred_pro_confidence_score, test_data_label_arr, pred_label_arr):
    judge_y = pred_pro_confidence_score - fit_func(pred_pro_weighted_sum, const.popt[label_en][0], const.popt[label_en][1], const.popt[label_en][2])
    above_fit_func_index = np.where(judge_y >= 0)[0]  # 当popt的保留小数位数较短时，up_data_index/down_data_index内元素的个数将产生个位数的偏差
    below_fit_func_index = np.where(judge_y < 0)[0]
    above_fit_func_predict_information = {
        "pred_pro_weighted_sum": pred_pro_weighted_sum[above_fit_func_index],
        "test_data_label_arr":   test_data_label_arr[above_fit_func_index],
        "pred_label_arr":        pred_label_arr[above_fit_func_index]
    }
    below_fit_func_predict_information = {
        "pred_pro_weighted_sum": pred_pro_weighted_sum[below_fit_func_index],
        "test_data_label_arr":   test_data_label_arr[below_fit_func_index],
        "pred_label_arr":        pred_label_arr[below_fit_func_index]
    }
    return above_fit_func_predict_information, below_fit_func_predict_information
def calc_accuracy(tp, tn, amount_sum):
    return (tp + tn) / amount_sum
def calc_ppv(tp, fp):
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)
def calc_tpr(tp, fn):
    return tp / (tp + fn)
def calc_f1(tp, fn, fp):
    return 2 * tp / (2 * tp + fn + fp)  # 2*ppv*tpr/(ppv+tpr)
def count_label_arr_and_calc_assessed_values(predict_information_dict):
    predict_positive_label_index = np.where(predict_information_dict["pred_label_arr"] == 1)[0]
    predict_negative_label_index = np.where(predict_information_dict["pred_label_arr"] == 0)[0]
    test_data_positive_label_index = np.where(predict_information_dict["test_data_label_arr"] == 1)[0]
    test_data_negative_label_index = np.where(predict_information_dict["test_data_label_arr"] == 0)[0]
    tp_index = list(set(predict_positive_label_index) & set(test_data_positive_label_index))  # 取交集
    fn_index = list(set(predict_negative_label_index) & set(test_data_positive_label_index))
    fp_index = list(set(predict_positive_label_index) & set(test_data_negative_label_index))
    tn_index = list(set(predict_negative_label_index) & set(test_data_negative_label_index))
    tp_num = len(tp_index)  # 正类标记，被预测成正类的个数
    fn_num = len(fn_index)  # 正类标记，被预测成负类的个数
    fp_num = len(fp_index)  # 负类标记，被预测成正类的个数
    tn_num = len(tn_index)  # 负类标记，被预测成负类的个数
    amount_sum = tp_num + fn_num + fp_num + tn_num  # 总数
    false_positive_rate, true_positive_rate, thresholds_list = \
        roc_curve(predict_information_dict["test_data_label_arr"], predict_information_dict["pred_pro_weighted_sum"], pos_label=1)
    calc_assessed_values = {
        "acc":    calc_accuracy(tp_num, tn_num, amount_sum),
        "ppv":    calc_ppv(tp_num, fp_num),
        "tpr":    calc_tpr(tp_num, fn_num),
        "f1":     calc_f1(tp_num, fn_num, fp_num),
        "au_roc": auc(false_positive_rate, true_positive_rate),
        "tp_num": tp_num, "fn_num": fn_num, "fp_num": fp_num, "tn_num": tn_num
    }
    return calc_assessed_values
def do_work():
    float_assessed_values_name_list = ["ACC", "PPV", "TPR", "F1", "AUC"]
    float_assessed_values_name_lowercase_list = ["acc", "ppv", "tpr", "f1", "au_roc"]
    int_assessed_values_name_list = ["TP", "FN", "FP", "TN"]
    int_assessed_values_name_lowercase_list = ["tp_num", "fn_num", "fp_num", "tn_num"]
    for label_en, label_ch in zip(const.label_en_list, const.label_ch_list):
        # 导入风险概率、置信得分、标记空间、预测结果
        pred_pro_weighted_sum, pred_pro_confidence_score, test_data_label_arr, pred_label_arr = prepare_predict_information(label_en, label_ch)
        # 区分高置信区间、低置信区间的风险概率、置信得分、标记空间、预测结果
        above_fit_func_predict_information, below_fit_func_predict_information =\
            prepare_predict_information_dict(label_en, pred_pro_weighted_sum, pred_pro_confidence_score, test_data_label_arr, pred_label_arr)
        # 评估高置信区间的预测效果
        calc_assessed_values_above_fit_func = count_label_arr_and_calc_assessed_values(above_fit_func_predict_information)
        # 评估低置信区间的预测效果
        calc_assessed_values_below_fit_func = count_label_arr_and_calc_assessed_values(below_fit_func_predict_information)
        # 输出评估
        print("\n%s\n\t高置信区间\t低置信区间" % label_ch)
        for name, name_lowercase in zip(float_assessed_values_name_list, float_assessed_values_name_lowercase_list):
            print("%s\t%.3f\t\t%.3f" % (name, calc_assessed_values_above_fit_func[name_lowercase], calc_assessed_values_below_fit_func[name_lowercase]))
        for name, name_lowercase in zip(int_assessed_values_name_list, int_assessed_values_name_lowercase_list):
            print("%s\t%d\t\t%d" % (name, calc_assessed_values_above_fit_func[name_lowercase], calc_assessed_values_below_fit_func[name_lowercase]))
if __name__ == "__main__":
    do_work()
