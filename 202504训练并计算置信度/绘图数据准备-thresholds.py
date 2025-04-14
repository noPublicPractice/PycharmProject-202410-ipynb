import joblib
import numpy as np
import json

class const:
    label = 'MI'  # AF,AS,MI,CI
    label_ch = '心梗'  # 房颤,动脉粥样硬化,心梗,脑梗
    folds = 6
    great_random_state = 12000
    model_name = 'LGBM'
    model_values_joblib_name = r'思路1：训练并计算置信度\模型数据pkl\%s模型pkl-Fold=%d-random_state=%d-exclude_Fold=%d-model_name=%s.pkl'
    do_write_json_path___ = r'..\PycharmProject-202302-绘图\集成学习绘图\%s%s_thresholds_dict.json'
def do_write_pic_data(pred_pro_list):
    pred_pro_arr = np.array(pred_pro_list)
    pred_pro_arr = pred_pro_arr.T
    pred_pro_mean = np.mean(pred_pro_arr, axis=1)  # 风险概率
    pred_pro_std = np.std(pred_pro_arr, axis=1)  # 置信得分
    thresholds_dict = {
        "pred_pro_mean": list(pred_pro_mean),
        "pred_pro_std":  list(1 - pred_pro_std)
    }
    fp = open(const.do_write_json_path___ % (const.label_ch, const.label), 'w', encoding='utf-8')
    json.dump(thresholds_dict, fp, indent=4, ensure_ascii=False, sort_keys=False)
    fp.close()
def enter_pkl_and_call_the_evaluation_function():
    pred_pro_list = []
    for fold in range(const.folds):
        now_model_values_joblib_name = const.model_values_joblib_name % (const.label, const.folds, const.great_random_state, fold + 1, const.model_name)
        vectorizer, model, test_data_label, pred_pro, pred_lr = joblib.load(open(now_model_values_joblib_name, "rb"))
        pred_pro_list.append(list(pred_pro))
    do_write_pic_data(pred_pro_list)
if __name__ == "__main__":
    enter_pkl_and_call_the_evaluation_function()
