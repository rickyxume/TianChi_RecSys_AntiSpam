

from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
import os
import pandas as pd
import numpy as np
from utils import read_data
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'


def fetch_pred(path):
    from sklearn.metrics import f1_score
    import json
    graph_meta = json.loads(open(path + '/graph_meta.json', 'r').read())
    graph_meta['output_names'] = ['output/Sigmoid:0']
    with tf.gfile.FastGFile(path + "/frozen_inference_graph.pb", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        try:
            x, result = tf.import_graph_def(graph_def, return_elements=list(
                map(lambda x: x[0], graph_meta.values())))

        except:
            graph_meta['output_names'] = ['output_1/Sigmoid:0']
            x, result = tf.import_graph_def(graph_def, return_elements=list(
                map(lambda x: x[0], graph_meta.values())))

    df_truth = read_data(path='/tcdata/truth.csv', nrows=1000000)
    df_test0 = read_data(path='/tcdata/predict0.csv', nrows=1000000)
    df_test1 = read_data(path='/tcdata/predict1.csv', nrows=1000000)
    df_test = pd.concat([df_test0, df_test1], axis=0).reset_index()
    df_test['label'] = df_truth.label
    df_test0 = df_test[:50000]
    df_test1 = df_test[50000:]

    x_fe = df_test0['features'].str.split(
        " ", expand=True).values.astype(np.float32)
    id_cols = ['user_id', 'item_id']
    x_id = df_test0[id_cols].values
    x_test0 = np.concatenate((x_id, x_fe), axis=1)
    y_test0 = df_test0['label'].values

    x_fe = df_test1['features'].str.split(
        " ", expand=True).values.astype(np.float32)
    id_cols = ['user_id', 'item_id']
    x_id = df_test1[id_cols].values
    x_test1 = np.concatenate((x_id, x_fe), axis=1)
    y_test1 = df_test1['label'].values

    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        pred0 = sess.run(result, feed_dict={x: x_test0})
        pred0 = pred0.flatten()
        pred1 = sess.run(result, feed_dict={x: x_test1})
        pred1 = pred1.flatten()
        return pred0, y_test0, pred1, y_test1


def search_thr(pred0, y_test0, pred1, y_test1, thr, callback=f1_score):
    # print('thr:', thr)
    # print(f"F1_0:{f1_score((pred0 > thr).astype(int), y_test0)}")
    # print(f"F1_1:{f1_score((pred1 > thr).astype(int), y_test1)}")
    return callback((pred0 > thr).astype(int), y_test0), callback((pred1 > thr).astype(int), y_test1)


def evaluate():
    model_dir = './model'
    save_name = 'saved_model'
    # train_path = '/tcdata/train0.csv'
    train_path = '/tcdata/train1.csv'
    # train(train_path, model_dir, save_name)
    # evaluate_score('./model/frozen_model', 0.18)
    pred0, y_test0, pred1, y_test1 = fetch_pred('./model/frozen_model')
    f11_lst = []
    f12_lst = []
    p11_lst = []
    p12_lst = []
    r11_lst = []
    r12_lst = []
    max_score = 0
    max_index = 0
    for i in range(1, 100):
        f11, f12 = search_thr(pred0, y_test0, pred1, y_test1, i / 100)
        p11, p12 = search_thr(pred0, y_test0, pred1,
                              y_test1, i / 100, precision_score)
        r11, r12 = search_thr(pred0, y_test0, pred1,
                              y_test1, i / 100, recall_score)
        score = f11 + f12
        if score > max_score:
            max_score = score
            max_index = i
        f11_lst.append(f11)
        f12_lst.append(f12)
        p11_lst.append(p11)
        p12_lst.append(p12)
        r11_lst.append(r11)
        r12_lst.append(r12)
    # plt.figure(figsize=(5,6))
    plt.ylim((0, 1.0))
    plt.xlabel('threshold')
    plt.ylabel('score')

    plt.plot(range(1, 100), f11_lst)
    plt.plot(range(1, 100), f12_lst)
    plt.plot(range(1, 100), p11_lst)
    plt.plot(range(1, 100), p12_lst)
    plt.plot(range(1, 100), r11_lst)
    plt.plot(range(1, 100), r12_lst)
    plt.axvline(max_index)
    sc = f"""'opt_thr': {max_index / 100:.4}, 'score': {max_score:.4}, 'F11': {f11_lst[max_index]:.4}, 'F12': {f12_lst[max_index]:.4},\n 'P1': {p11_lst[max_index]:.4}, 'P2': {p12_lst[max_index]:.4}, 'R1': {r11_lst[max_index]:.4}, 'R2': {r12_lst[max_index]:.4} """
    print(sc)
    # print(f"""&{np.mean(F11)*100:.2f}($\pm{np.std(F11)*100:.2f}$) &{np.mean(P1)*100:.2f}($\pm{np.std(P1)*100:.2f}$) &{np.mean(R1)*100:.2f}($\pm{np.std(R1)*100:.2f}$) &{np.mean(F12)*100:.2f}($\pm{np.std(F12)*100:.2f}$) &{np.mean(P2)*100:.2f}($\pm{np.std(P2)*100:.2f}$) &{np.mean(R2)*100:.2f}($\pm{np.std(R2)*100:.2f}$) &{np.mean(score)*100:.2f}($\pm{np.std(score)*100:.2f}$) """)
    plt.grid()
    plt.title(sc, fontsize=8)
    # plt.text(0, -0.15, sc)
    plt.show()


if __name__ == '__main__':
    evaluate()
