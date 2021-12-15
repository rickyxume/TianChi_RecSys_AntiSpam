import os
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import time
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


def evaluate_latency(result_path):
    df_pred = pd.read_csv(result_path, header=None, names=[
        'uuid', 'time_in', 'time_out', 'label'])
    df_pred = df_pred.sort_values(by='uuid')
    df_pred.index = range(df_pred.shape[0])
    latency = df_pred['time_out'] - df_pred['time_in']
    print(','.join(map(str, latency.values.tolist())))
    print(','.join(map(str, df_pred['label'].values.tolist())))

    df_pred0 = df_pred[:50000]
    df_pred1 = df_pred[50000:]

    time_diff0 = (df_pred0['time_out'] - df_pred0['time_in'])
    time_mask0 = time_diff0 <= 500
    valid_latency_0 = time_mask0.mean()

    time_diff1 = (df_pred1['time_out'] - df_pred1['time_in'])
    time_mask1 = time_diff1 <= 500
    valid_latency_1 = time_mask1.mean()

    print(f'第一阶段平均耗时: {time_diff0.mean()}ms')
    print(f'前200条平均耗时: {time_diff0[:200].mean()}ms')
    print(f'前1000条平均耗时: {time_diff0[:1000].mean()}ms')
    avg_time0 = time_diff0[:50000].mean()
    print(f'前50000条平均耗时: {avg_time0}ms')
    print(f'valid_latency_0: {valid_latency_0:.4f}')

    print(f'第二阶段平均耗时: {time_diff1.mean()}ms')
    print(f'前200条平均耗时: {time_diff1[:200].mean()}ms')
    print(f'前1000条平均耗时: {time_diff1[:1000].mean()}ms')
    avg_time1 = time_diff1[:50000].mean()
    print(f'前50000条平均耗时: {avg_time1}ms')
    print(f'valid_latency_1: {valid_latency_1:.4f}')

    print(f'总体平均耗时: {(avg_time0+avg_time1)/2:.4f}ms')


def evaluate(result_path, truth_path):
    df_pred = pd.read_csv(result_path, header=None, names=[
                          'uuid', 'time_in', 'time_out', 'label'])
    df_pred0 = df_pred[:50000]
    df_pred1 = df_pred[50000:]

    df_truth = pd.read_csv(truth_path, header=None, names=['uuid', 'label'])
    df_truth0 = df_truth[:50000]
    df_truth1 = df_truth[50000:]

    time_diff0 = (df_pred0['time_out'] - df_pred0['time_in'])
    time_mask0 = time_diff0 <= 500
    valid_latency_0 = time_mask0.mean()

    time_diff1 = (df_pred1['time_out'] - df_pred1['time_in'])
    time_mask1 = time_diff1 <= 500
    valid_latency_1 = time_mask1.mean()

    F1_0 = f1_score(df_truth0.label, df_pred0.label)
    F1_1 = f1_score(df_truth1.label, df_pred1.label)
    print(f'第一阶段平均耗时: {time_diff0.mean()}s')
    print(f'前200条平均耗时: {time_diff0[:200].mean()}s')
    print(f'前1000条平均耗时: {time_diff0[:1000].mean()}s')
    avg_time0 = time_diff0[:50000].mean()
    print(f'前50000条平均耗时: {avg_time0}ms')
    print(f'valid_latency_0: {valid_latency_0:.4f}')
    print(f'F1_0: {F1_0:.4f}')
    print(f'第二阶段平均耗时: {time_diff1.mean()}s')
    print(f'前200条平均耗时: {time_diff1[:200].mean()}s')
    print(f'前1000条平均耗时: {time_diff1[:1000].mean()}s')
    avg_time1 = time_diff1[:50000].mean()
    print(f'前50000条平均耗时: {avg_time1}ms')
    print(f'valid_latency_1: {valid_latency_1:.4f}')
    print(f'F1_1: {F1_1:.4f}')
    print(f'总体平均耗时: {(avg_time0+avg_time1)/2:.4f}ms')
    print(f"得分:{F1_0*valid_latency_0 + F1_1*valid_latency_1}")

    # 可视化
    grid = gridspec.GridSpec(1, 2)
    plt.figure(figsize=(16, 4))
    for n, latency in enumerate([time_diff0, time_diff1]):
        ax = plt.subplot(grid[n])
        ax.plot(latency, color='g')
        if n < 1:
            ax.set_ylabel('Latency(ms)')
        ax.set_xlabel('Message ID')
        ax.set_title(f"Stage {str(n)}")
    plt.savefig("./latency.png")


if __name__ == '__main__':
    # result_path = '/root/tianchi_entry/result.csv'
    result_path = './result.csv'
    truth_path = '/tcdata/truth.csv'
    evaluate(result_path, truth_path)
    # evaluate_latency(result_path)
