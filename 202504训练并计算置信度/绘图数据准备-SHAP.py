import pickle
import joblib
import shap
import json

class const:
    max_features_num = 2000
    label = 'CI'  # AF,AS,MI,CI
    label_ch = '脑梗'  # 房颤,动脉粥样硬化,心梗,脑梗
    folds = 6
    great_random_state = 15000
    model_name = 'LGBM'
    data_frame_pickle_pat = r'思路1：训练并计算置信度\交叉训练集csv、pickle\%s样本空间、标记空间-Fold=%d-random_state=%d-exclude_Fold=%d.pickle'
    model_values_pkl_name = r'思路1：训练并计算置信度\模型数据pkl\%s模型pkl-Fold=%d-random_state=%d-exclude_Fold=%d-model_name=%s.pkl'
    do_write_json_path___ = r'..\PycharmProject-202302-绘图\集成学习绘图\%s%s_shap_dict.json'
def do_shap(exclude_fold, model, train_data_observations, test_data_observations, feature_names):
    X_train_dense = train_data_observations.toarray()
    X_test_dense = test_data_observations.toarray()
    explainer = shap.Explainer(model, X_train_dense)  # 创建 SHAP Explainer 对象
    for i in range(0, 1):
        predicted_proba = model.predict_proba(X_test_dense[i].reshape(1, -1))[0, 1]
        print(f"Predicted Probability for Sample {i}: {predicted_proba:.2f}")  # 输出预测的概率值
        shap_values = explainer(X_test_dense[i].reshape(1, -1))  # 选择一个样本进行解释
        shap_values_single = shap_values[0]  # 获取样本的 SHAP 值
        shap_dict = {
            "predicted_proba": predicted_proba,
            "values":          list(shap_values_single.values),
            "base_values":     float(shap_values_single.base_values),
            "data":            list(X_test_dense[i].astype(float)),
            "feature_names":   list(feature_names)
        }
        fp = open(const.do_write_json_path___ % (const.label_ch, const.label), 'w', encoding='utf-8')
        json.dump(shap_dict, fp, indent=4, ensure_ascii=False, sort_keys=False)
        fp.close()
def enter_pkl_and_call_the_evaluation_function():
    for fold in range(0, 1):  # (const.folds):
        now_model_values_pkl_name = const.model_values_pkl_name % (const.label, const.folds, const.great_random_state, fold + 1, const.model_name)
        vectorizer, model, test_data_label, pred_pro, pred_lr = joblib.load(open(now_model_values_pkl_name, "rb"))
        pickle_path = const.data_frame_pickle_pat % (const.label, const.folds, const.great_random_state, fold + 1)
        train_data_frame, test_data_frame = pickle.load(open(pickle_path, "rb"))  # 从文件中恢复变量.注意:二进制模式不采用编码参数
        train_data_observations = vectorizer.fit_transform(train_data_frame['观察值'])
        test_data_observations = vectorizer.transform(test_data_frame['观察值'])
        feature_names = vectorizer.get_feature_names_out()  # 获取特征名称
        do_shap(fold + 1, model, train_data_observations, test_data_observations, feature_names)
if __name__ == "__main__":
    enter_pkl_and_call_the_evaluation_function()
