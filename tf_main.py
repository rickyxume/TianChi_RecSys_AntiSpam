import os
import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from Models import seed_tensorflow, freeze_model, TimingEarlyStoppingCheckpoint, batchwise_avg_f1
from utils import reduce_mem_usage, get_optimal_Fscore, read_data, split_labeled
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Model
from tensorflow.keras.layers import add, Input, Dense, Dropout, ReLU, Concatenate, BatchNormalization
import warnings
warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


def build_model(inp_dim, hidden_units=[128, 32, 8], deep_dropout_rate=0.3, use_deep_res=True):
    from Models import SplitUserItemId
    input = Input((inp_dim,), name='input')
    ids, item_features, user_features, u80 = SplitUserItemId(
        name='Split_Layer')(input)
    features = Concatenate()([user_features, item_features, u80])
    deep = features
    for units in hidden_units:
        deep = Dense(units, use_bias=False)(deep)
        deep = BatchNormalization()(deep)
        deep = Dropout(deep_dropout_rate)(deep)
        deep = ReLU()(deep)
        if use_deep_res:
            deep_res = Dropout(deep_dropout_rate)(deep)
            deep = add([deep_res, deep])
    output = Dense(1, activation="sigmoid", name='output')(deep)
    model = Model(input, output)
    model.summary()
    return model


def train(train_path, model_dir, save_name):
    time_start = time.time()
    nrows = 10e5
    SEED = 20200709
    """数据处理策略"""
    test_size = 0.5  # 划分的验证集占比
    shuffle = True  # 是否打乱数据集
    """训练策略"""
    lr = 0.001
    batch_size = 200
    epochs = 200
    lr_patience = 6  # 学习率平原衰减的忍耐轮数,每次减半
    es_patience = 10  # 训练早停的忍耐轮数
    max_training_time = 70  # 最长训练时间(单位min)
    min_saving_epoch = 20  # 保存checkpoint的开始轮数
    """模型参数"""
    hidden_units = [256, 128, 32, 8]  # 各层神经元数
    deep_dropout_rate = 0.25  # dropout比率
    use_deep_res = True  # 使用dropout残差
    """focal loss参数"""
    ALPHA = 0.75
    GAMMA = 2.0

    """输入"""
    inp_dim = 154

    seed_tensorflow(SEED)
    print(f"处理{nrows}条数据")
    data = read_data(path=train_path, nrows=nrows)
    if train_path == '/tcdata/train1.csv':
        print("第二阶段")  # 更新数据
        try:
            data0 = read_data(path='/tcdata/train0.csv', nrows=nrows)
            df_test0 = read_data(path='/tcdata/predict0.csv', nrows=1000000)
            df_test0['label'] = -1
            data = pd.concat([data, data0, df_test0], axis=0)
        except:
            pass
    from utils import label_user_item_via_blacklist, label_data_via_blacklist
    print(f"{data.label.value_counts()}")
    data = label_user_item_via_blacklist(data)
    data = label_data_via_blacklist(data)
    print(f"{data.label.value_counts()}")
    # data = label_user_item_via_blacklist(data)
    # data = label_data_via_blacklist(data)
    # print(f"{data.label.value_counts()}")

    data = reduce_mem_usage(data)

    data_labeled, data_no_labeled = split_labeled(data)
    print(f"处理有标签数据,{data_labeled.shape}")

    # 有标签的重复的数据处理一下还能用来做数据增广
    # from utils import rand_mask
    # data_dup = data_labeled[data_labeled.duplicated(['user_id', 'item_id'])]
    # x_fe = data_dup['features'].str.split(
    #     " ", expand=True).values.astype(np.float32)
    # id_cols = ['user_id', 'item_id']
    # x_id = data_dup[id_cols].values
    # y_dup = data_dup['label'].values
    # x_dup = np.concatenate((x_id, x_fe), axis=1)
    # x_dup_new = np.apply_along_axis(lambda x: rand_mask(x), axis=1, arr=x_dup)
    # print(f"重复利用有标签的重复样本,{x_dup_new.shape}")

    print("去重")
    data_labeled.drop_duplicates(
        subset=['user_id', 'item_id'], keep='first', inplace=True)
    x_fe = data_labeled['features'].str.split(
        " ", expand=True).values.astype(np.float32)
    # concat id 方便后面做embedding
    id_cols = ['user_id', 'item_id']
    x_id = data_labeled[id_cols].values
    x = np.concatenate((x_id, x_fe), axis=1)
    y = data_labeled['label'].values
    print(f"原始无重复样本,{x.shape}")

    model = build_model(inp_dim=inp_dim, hidden_units=hidden_units,
                        deep_dropout_rate=deep_dropout_rate, use_deep_res=use_deep_res)

    optimizer = Adam(learning_rate=lr)
    metrics = [
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        tf.keras.metrics.AUC(name='auprc', curve='PR'),
        batchwise_avg_f1
    ]

    def FocalLoss(y_true, y_pred, alpha=ALPHA, gamma=GAMMA):
        BCE = binary_crossentropy(y_true, y_pred)
        BCE_EXP = K.exp(-BCE)
        focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)
        return focal_loss

    model.compile(optimizer=optimizer,
                  loss=FocalLoss,
                  metrics=metrics)
    save_path = os.path.join(model_dir, 'frozen_model')
    os.makedirs(save_path, exist_ok=True)
    best_model_path = os.path.normpath(
        os.path.join(model_dir, save_name+'.h5'))
    if train_path == '/tcdata/train1.csv':
        print("第二阶段")  # 线上失败了，所以没怎么调这个
        try:
            model.load_weights(best_model_path)
            print('加载第一轮训练的best model')
            min_saving_epoch = 10  # 训到第10轮再开始保存模型
            test_size = 0.7
            max_training_time = 11  # 限时训练 11min
        except:
            print('第一轮训练的best model加载失败')
    tesckp = TimingEarlyStoppingCheckpoint(best_model_path, monitor="val_auc",
                                           save_best_only=True, verbose=2, mode='max',
                                           max_training_time=max_training_time, min_saving_epoch=min_saving_epoch)
    es = EarlyStopping(monitor='val_loss', patience=es_patience, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                  patience=lr_patience, verbose=1, mode='auto',
                                  epsilon=1e-6, cooldown=1, min_lr=1e-7)

    x_trn, x_val, y_trn, y_val = train_test_split(
        x, y, test_size=test_size, shuffle=shuffle, random_state=SEED)
    # x_train = np.concatenate((x_trn, x_dup_new), axis=0)
    # y_train = np.concatenate((y_trn, y_dup), axis=0)
    x_train = x_trn
    y_train = y_trn
    print(f"x_train:{x_train.shape},y_train:{Counter(y_train)}")
    print(f"x_val:{x_val.shape},y_val:{Counter(y_val)}")

    print("开始训练模型")
    if train_path == '/tcdata/train0.csv':
        print("第一阶段训练")
        tesckp_first = TimingEarlyStoppingCheckpoint(best_model_path, monitor="val_auc",
                                                     save_best_only=True, verbose=2, mode='max',
                                                     max_training_time=max_training_time, min_saving_epoch=min_saving_epoch)
        es_first = EarlyStopping(
            monitor='val_loss', patience=es_patience, mode='min')
        reduce_lr_first = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                            patience=lr_patience, verbose=1, mode='auto',
                                            epsilon=1e-6, cooldown=1, min_lr=1e-7)
        print("先使用验证集训练")
        model.fit(x_val, y_val,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=[es_first, tesckp_first, reduce_lr_first],
                  validation_data=(x_train, y_train),
                  )
        try:
            model.load_weights(best_model_path)
            print('Load best model succeed')
        except:
            print('Load best model failed')

    print("使用训练集训练")
    model.fit(x_train, y_train,
              epochs=epochs,
              batch_size=batch_size,
              callbacks=[es, tesckp, reduce_lr],
              validation_data=(x_val, y_val),
              )
    try:
        model.load_weights(best_model_path)
        print('Load best model succeed')
    except:
        print('Load best model failed')

    # 阈值搜索
    opt_thr, opt_f1 = get_optimal_Fscore(model, x_val, y_val)
    print(f"only val -> opt_thr:{opt_thr},opt_f1:{opt_f1}")
    # 追加阈值层
    output = tf.cast(model.output > opt_thr, tf.int32, name='pred_label')
    model = Model(model.input, output)
    # 冻结模型
    freeze_model(save_path, K.get_session(),
                 input_names=[inp.op.name for inp in model.inputs],
                 output_names=[out.op.name for out in model.outputs])
    print(f'Freeze best model with threshold: {opt_thr}')
    cost_time = time.time() - time_start
    print(f"train耗时:{cost_time}s")


if __name__ == '__main__':
    from evaluation import evaluate
    try:
        import shutil
        shutil.rmtree('./model')
        print('removed')
    except:
        print('No such files')
    model_dir = './model'
    save_name = 'saved_model'
    train_path = '/tcdata/train0.csv'
    train(train_path, model_dir, save_name)
    evaluate()
    # train_path = '/tcdata/train1.csv'
    # train(train_path, model_dir, save_name)
    # evaluate()
