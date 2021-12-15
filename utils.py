from collections import defaultdict
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import f1_score


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def split_labeled(data):
    is_labeled = (data['label'] != -1)
    return data[is_labeled], data[~is_labeled]


# def split_dataset(raw_dataset_path, new_dataset_path):
#     # 主要是方便EDA
#     item_cols = [f'i{i}' for i in range(1, 72+1)]
#     user_cols = [f'u{i}' for i in range(1, 80+1)]
#     try:
#         with open(raw_dataset_path, 'r', encoding='utf-8') as rf:
#             with open(new_dataset_path, 'w+', encoding='utf-8') as wf:
#                 if "train" in raw_dataset_path:
#                     header = f"""uuid,visit_time,user_id,item_id,{str(item_cols+user_cols)[2:-2].replace("'", "").replace(" ","")},label"""
#                 else:  # "predict"
#                     header = f"""uuid,visit_time,user_id,item_id,{str(item_cols+user_cols)[2:-2].replace("'", "").replace(" ","")}"""
#                 wf.write(header+'\n')
#                 for line in rf:
#                     if "features" in line:
#                         continue
#                     line = str(line[:].split(" ")).replace("'", "")[1:-3]
#                     wf.write(line+'\n')
#     except FileNotFoundError:
#         print(f'{raw_dataset_path} 文件不存在！')


# def read_split_data(path, nrows=1000000):
#     df_chunk = pd.read_csv(path, chunksize=1e6, iterator=True, nrows=nrows)
#     data = pd.concat([chunk for chunk in df_chunk])
#     data = reduce_mem_usage(data)
#     return data


def read_data(path='/tcdata/train0.csv', nrows=1000000):
    if "train" in path:
        df_chunk = pd.read_csv(path, chunksize=1e6, iterator=True,
                               names=["uuid", "visit_time", "user_id", "item_id", "features", "label"], nrows=nrows)
        data = pd.concat([chunk for chunk in df_chunk])
        data = reduce_mem_usage(data)
    elif "predict" in path:
        df_chunk = pd.read_csv(path, chunksize=5e5, iterator=True,
                               names=["uuid", "visit_time", "user_id", "item_id", "features"], nrows=nrows)
        data = pd.concat([chunk for chunk in df_chunk])
        data = reduce_mem_usage(data)
    else:  # "truth"
        data = pd.read_csv(path, names=["uuid", "label"], nrows=nrows)

    return data


def label_user_item_via_blacklist(data):
    data_labeled, data_no_labeled = split_labeled(data)
    data_spam = data_labeled[data_labeled.label == 1]
    data_norm = data_labeled[data_labeled.label == 0]
    try:
        user_spam_dict = load_obj("user_black_dict")
        item_spam_dict = load_obj("item_black_dict")
        print("更新 user 和 item 黑名单")
    except:
        user_spam_dict = defaultdict(int)
        item_spam_dict = defaultdict(int)
        print("新建 user 和 item 黑名单")
    for _, row in data_spam[['user_id', 'item_id']].iterrows():
        u, i = row['user_id'], row['item_id']
        user_spam_dict[u] += 1  # 记录次数
        item_spam_dict[i] += 1  # 记录次数
    save_obj(user_spam_dict, "user_black_dict")
    save_obj(item_spam_dict, "item_black_dict")

    # 1、根据label=1确定绝对无误的用户黑名单和商品黑名单
    # 2、根据label=0 以及用户黑名单 确定当前用户是恶意的 则当前商品是正常的，将当前商品更新进商品白名单
    #    根据label=0 以及商品黑名单 确定当前商品是恶意的 则当前用户是正常的，将当前用户更新进用户白名单
    # 3、根据用户白名单 以及label=0 确定当前用户是正常的 则当前商品是（正常或潜在恶意的）
    #    根据商品白名单 以及label=0 确定当前商品是正常的 则当前用户是（正常或潜在恶意的）
    # 4、根据label=-1 以及 更新完毕的黑白名单 确定用户和商品的标签
    # 可以忽略步骤3
    try:
        user_norm_dict = load_obj("user_white_dict")
        item_norm_dict = load_obj("item_white_dict")
        print("更新 user 和 item 白名单")
    except:
        user_norm_dict = defaultdict(int)
        item_norm_dict = defaultdict(int)
        print("新建 user 和 item 白名单")
    for _, row in data_norm[['user_id', 'item_id']].iterrows():
        u, i = row['user_id'], row['item_id']
        if i in item_spam_dict.keys():  # 如果当前商品是恶意的
            user_norm_dict[u] = 0  # 用户则是正常的，加入白名单
        # else: #当前商品可能正常或潜在恶意
        if u in user_spam_dict.keys():  # 如果当前用户是恶意的
            item_norm_dict[i] = 0  # 商品则是正常的，加入白名单
        # else: #当前用户可能正常或潜在恶意
        # user_unknown_dict[u] = 0  #潜在的
    save_obj(user_norm_dict, "user_white_dict")
    save_obj(item_norm_dict, "item_white_dict")

    print("基于黑名单和白名单，给未知样本打上标签")

    def black_white_dict(ui, black_dict, white_dict):
        if ui in black_dict.keys():
            return 1
        elif ui in white_dict.keys():
            return 0
        else:
            return -1

    data_no_labeled['user_label'] = data_no_labeled['user_id'].apply(
        lambda u: black_white_dict(u, user_spam_dict, user_norm_dict))
    data_no_labeled['item_label'] = data_no_labeled['item_id'].apply(
        lambda i: black_white_dict(i, item_spam_dict, item_norm_dict))

    def ui_label2label(u, i):
        if u == 1 and i == 1:
            return 1
        elif ((u == 1 and i == 0) or (u == 0 and i == 1) or (u == 0 and i == 0)):
            return 0
        else:
            return -1

    data_no_labeled['label'] = list(map(lambda u, i: ui_label2label(
        u, i), data_no_labeled['user_label'], data_no_labeled['item_label']))

    data_labeled['user_label'] = data_labeled['user_id'].apply(
        lambda u: black_white_dict(u, user_spam_dict, user_norm_dict))
    data_labeled['item_label'] = data_labeled['item_id'].apply(
        lambda i: black_white_dict(i, item_spam_dict, item_norm_dict))
    data = pd.concat([data_no_labeled, data_labeled], axis=0)
    return data


def label_data_via_blacklist(data):
    data_labeled, data_no_labeled = split_labeled(data)
    data_spam = data_labeled[data_labeled.label == 1]
    data_norm = data_labeled[data_labeled.label == 0]
    try:
        ui_spam_dict = load_obj("user_item_black_dict")
        print("更新 user-item 黑名单")
    except:
        ui_spam_dict = defaultdict(int)
        print("新建 user-item 黑名单")
    for _, row in data_spam[['user_id', 'item_id']].iterrows():
        ui = (row['user_id'], row['item_id'])
        ui_spam_dict[ui] += 1  # 记录次数
    save_obj(ui_spam_dict, "user_item_black_dict")

    try:
        ui_norm_dict = load_obj("user_item_white_dict")
        print("更新 user-item 白名单")
    except:
        ui_norm_dict = defaultdict(int)
        print("新建 user-item 白名单")
    for idx, row in data_norm[['user_id', 'item_id']].iterrows():
        ui = (row['user_id'], row['item_id'])
        ui_norm_dict[ui] = 0
    save_obj(ui_norm_dict, "user_item_white_dict")

    def black_white_list(ui, ui_spam_dict, ui_norm_dict):
        if ui in ui_spam_dict.keys():
            return 1
        elif ui in ui_norm_dict.keys():
            return 0
        else:
            return -1

    print("基于<user_id,item_id>设置黑白名单，打上伪标签")
    data_no_labeled['label'] = list(map(lambda u, i: black_white_list(
        (u, i), ui_spam_dict, ui_norm_dict), data_no_labeled['user_id'], data_no_labeled['item_id']))
    # data_pseudo = data_no_labeled[data_no_labeled.label != -1]
    # data_labeled = pd.concat([data_pseudo, data_labeled], axis=0)
    data = pd.concat([data_no_labeled, data_labeled], axis=0)
    return data


def rand_mask(x, p=0.1):
    # 保留id，剩下部分按概率p随机mask掉一部分特征
    ids_mask = [True, True]
    ids_mask.extend(np.random.rand(152) > p)
    return x * np.array(ids_mask)


def evaluate_score(res_csv_path, truth_csv_path):
    # "/root/tianchi_entry/result.csv"
    df_pred = pd.read_csv(res_csv_path, names=[
                          'uuid', 'time_in', 'time_out', 'pred'])
    df_truth = pd.read_csv(truth_csv_path, names=['uuid', 'label'])
    time_diff = (df_pred['time_out'] - df_pred['time_in'])
    time_mask = time_diff <= 500
    f1 = f1_score(df_truth['label'][time_mask], df_pred['pred'][time_mask])
    ratio = time_mask.mean()
    print(f'avg time: {time_diff.mean()}')
    print(f'f1 score: {f1}')
    print(f'ratio   : {ratio}')
    print(f'score   : {f1 * ratio}')


def find_best_threshold(y_true, y_pred, l=0.1, r=0.6, p=0.01):
    thresholds = np.arange(l, r, p)
    print(f"以精度为{p}在[{thresholds[0]},{thresholds[-1]}]范围内搜索F1最佳阈值", end=">>")
    fscore = np.zeros(shape=(len(thresholds)))
    for index, elem in enumerate(thresholds):
        thr2sub = np.vectorize(lambda x: 1 if x > elem else 0)
        y_preds = thr2sub(y_pred)
        fscore[index] = f1_score(y_true, y_preds)
    index = np.argmax(fscore)
    thresholdOpt = thresholds[index]
    fscoreOpt = round(fscore[index], ndigits=8)
    print(f'最佳阈值:={thresholdOpt}->F1={fscoreOpt}')
    return thresholdOpt, fscoreOpt


def get_optimal_Fscore(model, x, y):
    # 由粗到细查找会比较快
    pred = model.predict(x)
    p = 0.1
    thr, best_fscore = find_best_threshold(y, pred, 0.1, 0.9, p)
    p /= 5
    thr, best_fscore = find_best_threshold(
        y, pred, round(thr-0.1, 4), round(thr+0.1, 4), p)
    p /= 5
    thr_optimal, best_fscore = find_best_threshold(
        y, pred, round(thr - 0.05, 4), round(thr + 0.05, 4), p)
    return thr_optimal, best_fscore


def reduce_mem_usage(df):
    # start_mem = df.memory_usage().sum()
    # print(f'压缩内存>>{start_mem:.2f}', end="->")
    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    # end_mem = df.memory_usage().sum()
    # print(f'{end_mem:.2f} MB', end=" ")
    # print(f'- {(100*(start_mem - end_mem) / start_mem):.1f}%', end=" -> ")
    return df


def plot_model_history_curve(model_history):
    import matplotlib.pyplot as plt
    plt.plot(model_history.history['auprc'])
    plt.plot(model_history.history['val_auprc'])
    plt.title('model auprc')
    plt.ylabel('auprc')
    plt.xlabel('epoch')
    plt.legend(['train-auprc', 'val-auprc'], loc='best')
    plt.show()

    plt.plot(model_history.history['loss'])
    plt.plot(model_history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train-loss', 'val-loss'], loc='best')
    plt.show()

    plt.plot(model_history.history['auc'])
    plt.plot(model_history.history['val_auc'])
    plt.title('model auc')
    plt.ylabel('auc')
    plt.xlabel('epoch')
    plt.legend(['train-auc', 'val-auc'], loc='best')
    plt.show()

    plt.plot(model_history.history['batchwise_avg_f1'])
    plt.plot(model_history.history['val_batchwise_avg_f1'])
    plt.title('model batchwise_avg_f1')
    plt.ylabel('batchwise_avg_f1')
    plt.xlabel('epoch')
    plt.legend(['train-f1', 'val-f1'], loc='best')
    plt.show()
