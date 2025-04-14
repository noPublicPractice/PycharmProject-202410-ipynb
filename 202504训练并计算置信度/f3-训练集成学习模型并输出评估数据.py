from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer  # 特征提取：使用TfidfVectorizer将观测值转换为特征
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
import time
import joblib
from sklearn.metrics import roc_curve, auc, recall_score
import pandas as pd

class const:
    max_features_num = 2000
    label = 'CI'  # AF,AS,MI,CI
    label_ch = '脑梗'  # 房颤,动脉粥样硬化,心梗,脑梗
    folds = 6
    great_random_state = 15000
    model_name = 'LGBM'
    data_frame_pickle_pat = r'交叉训练集csv、pickle\%s样本空间、标记空间-Fold=%d-random_state=%d-exclude_Fold=%d.pickle'
    model_values_pkl_name = r'模型数据pkl\%s模型pkl-Fold=%d-random_state=%d-exclude_Fold=%d-model_name=%s.pkl'
    do_write_file_path___ = 'f4-最终的模型评估数据.csv'
    temp_time = 0.0
def cal_time(value_name):
    print("%s计时:%f秒" % (value_name, time.time() - const.temp_time))
    const.temp_time = time.time()
def get_lgbm_model():
    const.model_name = 'LGBM'
    return LGBMClassifier()  # 使用LGBM模型
def get_gbdt_model():
    const.model_name = 'GBDT'
    return GradientBoostingClassifier()
def get_xgbt_model():
    const.model_name = 'XGBt'
    return XGBClassifier()
def get_important_feature(vectorizer, model, values_num):
    feature_names = vectorizer.get_feature_names_out()  # 获取特征名称
    feature_importance = model.feature_importances_  # 获取特征重要性
    explaining_df = pd.DataFrame({"feature_names": feature_names, "得分": feature_importance})
    explaining_df_sort = explaining_df.sort_values(by="得分", ascending=False)  # 按重要性排序
    explaining_df_sort_values = explaining_df_sort.values
    if const.model_name in ['GBDT', 'XGBt']:
        for i in range(values_num):
            explaining_df_sort_values[i][1] = round(explaining_df_sort_values[i][1], 5)
    return str(explaining_df_sort_values[0:values_num]).replace('\n', '').replace('\'', '')  # print(explaining_df_sort[:20])
def do_write_evaluation(exclude_fold, vectorizer, model, test_data_label, pred_pro, pred_lr):
    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_data_label, pred_pro)
    au_roc = round(auc(false_positive_rate, true_positive_rate), 6)  # discrimination判别
    sensitivity = round(recall_score(test_data_label, pred_lr), 6)
    specificity = round(1 - recall_score(1 - test_data_label, pred_lr), 6)
    # 笔记：false_positive_rate = 假正例率 = FPR = FP/(FP+TN) = FP/所有实际是反例的
    # 笔记：true_positive_rate  = 真正例率 = TPR = TP/(TP+FN) = TP/所有实际是正例的 = 敏感性 = sensitivity
    # 笔记：                      1 - FP/(TN+FP) = TN/(FP+TN) = TN/所有实际是负例的 = 特异性 = specificity
    # 笔记：TP,预测是正样本,预测对了，FN,预测是负样本,实际是正样本，TN,预测是负样本,预测对了，FP,预测是正样本,实际是负样本
    # 笔记：metrics.recall_score()计算正样本召回率，浮点型
    do_write_tuple = (const.great_random_state, exclude_fold, const.label, sensitivity, specificity, discrimination, get_important_feature(vectorizer, model, 4))
    with open(const.do_write_file_path___, 'a', encoding='utf-8') as do_write_file_process:
        do_write_file_process.write('%d\t\t\t%d\t\t%s\t%f\t\t%f\t\t%f\t%s\n' % do_write_tuple)
def create_model():
    vectorizer = TfidfVectorizer(analyzer="word", max_features=const.max_features_num, dtype=np.float32)  # , max_df=0.95, min_df=0.05
    for fold in range(const.folds):
        ## 此处修改调用模型的函数
        model = get_lgbm_model()
        ## 判断pkl文件是否已生成
        now_model_values_pkl_name = const.model_values_pkl_name % (const.label, const.folds, const.great_random_state, fold + 1, const.model_name)
        if os.path.exists(now_model_values_pkl_name):
            continue  # 如果模型已存在
        ## 导入data_frame
        pickle_path = const.data_frame_pickle_pat % (const.label, const.folds, const.great_random_state, fold + 1)
        train_data_frame, test_data_frame = pickle.load(open(pickle_path, "rb"))  # 从文件中恢复变量.注意:二进制模式不采用编码参数
        ## 提取训练集、测试集的样本空间
        train_data_observations = vectorizer.fit_transform(train_data_frame['观察值'])
        test_data_observations = vectorizer.transform(test_data_frame['观察值'])
        ## 提取训练集、测试集的标记空间
        label_encoder = LabelEncoder()
        train_data_label = label_encoder.fit_transform(train_data_frame[const.label_ch])
        test_data_label = label_encoder.fit_transform(test_data_frame[const.label_ch])
        ## 训练操作
        const.temp_time = time.time()
        model.fit(train_data_observations, train_data_label)  # 时间消耗尺度：LGBM是1.048910秒，GBDT是54.389159秒，XGBt是
        cal_time('model')
        pred_pro = model.predict_proba(test_data_observations)[:, 1]  # 时间消耗尺度：不足0.1秒
        pred_lr = model.predict(test_data_observations)  # 时间消耗尺度：不足0.1秒
        # conf_mat = confusion_matrix(test_data_label, pred_lr)  # conf_mat即(confusion matrix)混淆矩阵，详见周志华《机器学习》2.3.2查准率、查全率与Fl
        ## 保存模型变量，空间消耗尺度占比--2:347:?:634:634（单位:KB）
        joblib.dump((vectorizer, model, test_data_label, pred_pro, pred_lr), open(now_model_values_pkl_name, "wb"))
        ## 将模型评估输出到文件
        do_write_evaluation(fold + 1, vectorizer, model, test_data_label, pred_pro, pred_lr)
def enter_pkl_and_call_the_evaluation_function():
    for fold in range(const.folds):
        now_model_values_pkl_name = const.model_values_pkl_name % (const.label, const.folds, const.great_random_state, fold + 1, const.model_name)
        if os.path.exists(now_model_values_pkl_name):
            vectorizer, model, test_data_label, pred_pro, pred_lr = joblib.load(open(now_model_values_pkl_name, "rb"))
            do_write_evaluation(fold + 1, vectorizer, model, test_data_label, pred_pro, pred_lr)
if __name__ == "__main__":
    create_model()
    # enter_pkl_and_call_the_evaluation_function()
