import tensorflow.keras.backend as K
import tensorflow as tf
from tensorflow.keras.layers import Layer, Input, Dense, Dropout, ReLU, Concatenate, Embedding, BatchNormalization, Activation, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Model, activations
from tensorflow.keras.regularizers import l2

import os
import time
import random
import numpy as np


def seed_tensorflow(seed=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 使用CPU
    # 固定随机种子
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    try:
        tf.compat.v1.set_random_seed(seed)
        # tf.set_random_seed(seed)
        tf.random.set_seed(seed)
    except:
        pass


class TimingEarlyStoppingCheckpoint(ModelCheckpoint):
    # 限时早停、指定轮数开始存ckp、
    def __init__(self,
                 filepath,
                 monitor='val_loss',
                 verbose=0,
                 save_best_only=False,
                 save_weights_only=False,
                 mode='auto',
                 save_freq='epoch',
                 max_training_time=0,
                 min_saving_epoch=20,
                 **kwargs):
        super(TimingEarlyStoppingCheckpoint, self).__init__(filepath,
                                                            monitor=monitor,
                                                            verbose=verbose,
                                                            save_best_only=save_best_only,
                                                            save_weights_only=save_weights_only,
                                                            mode=mode,
                                                            save_freq=save_freq,
                                                            **kwargs)
        self.max_training_time = max_training_time * 60
        self.min_saving_epoch = min_saving_epoch
        self.start_time = time.time()
        self.stopped_epoch = 0

    def on_train_begin(self, logs=None):
        self.start_time = time.time()
        self.stopped_epoch = 0
        super(TimingEarlyStoppingCheckpoint, self).on_train_begin(logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        if isinstance(self.save_freq, int):
            self._samples_seen_since_last_saving += logs.get('size', 1)
            if self._samples_seen_since_last_saving >= self.save_freq and self._current_epoch >= self.min_saving_epoch:
                self._save_model(epoch=self._current_epoch, logs=logs)
                self._samples_seen_since_last_saving = 0

    def on_epoch_end(self, epoch, logs=None):
        current = time.time()
        if current - self.start_time < self.max_training_time:
            if self.verbose > 0:
                print('\nEpoch %05d: current elapsed time: %fs' %
                      (epoch, current - self.start_time))
            self.epochs_since_last_save += 1
            if self.save_freq == 'epoch':
                if self.model._in_multi_worker_mode():
                    if self._current_epoch >= self.min_saving_epoch:
                        with self._training_state.untrack_vars():
                            self._save_model(epoch=epoch, logs=logs)
                else:
                    if self._current_epoch >= self.min_saving_epoch:
                        self._save_model(epoch=epoch, logs=logs)
            if self.model._in_multi_worker_mode():
                self._training_state.back_up(epoch)
        else:
            self.stopped_epoch = epoch
            self.model.stop_training = True


def batchwise_avg_f1(y_true, y_pred):
    """
    batch-wise average F1
    """
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


def freeze_model(save_path, session, keep_var_names=None, input_names=None, output_names=None, clear_devices=True):
    # https://github.com/pranayanand123/Keras-Model-to-tensorflow-frozen-.pb/blob/master/keras_to_tensorflow.py
    import json
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(
            set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        tf.train.write_graph(frozen_graph, save_path,
                             'frozen_inference_graph.pb', as_text=False)
        print(f"保存frozen_inference_graph.pb->{save_path}")
        meta = {
            "input_names": [i + ':0' for i in input_names],
            "output_names": [o + ':0' for o in output_names]
        }
        with open(os.path.join(save_path, "graph_meta.json"), "w") as f:
            f.write(json.dumps(meta))
            print("输入输出层名 -> graph_meta.json")


class SplitUserItemId(Layer):
    # 加入id，忽略无效特征
    def __init__(self, **kwargs):
        super(SplitUserItemId, self).__init__(**kwargs)

    def call(self, inputs):
        # 按第二个维度对tensor进行切片，返回一个list
        in_dim = K.int_shape(inputs)[-1]
        print(f"输入维度:{in_dim}")
        assert in_dim == 154
        # 忽略掉151,152,153
        return [inputs[:, :2], inputs[:, 2:2+72], inputs[:, 74:74+76], inputs[:, 153:154]]

    def compute_output_shape(self, input_shape):
        # output_shape也要是对应的list
        # in_dim = input_shape[-1]
        # user_id,item_id,user_features,item_features,label强相关的item_feature
        return [(None, 2), (None, 72), (None, 76), (None, 1)]


class SplitUI(Layer):
    # 忽略无效特征
    def __init__(self, **kwargs):
        super(SplitUI, self).__init__(**kwargs)

    def call(self, inputs):
        # 按第二个维度对tensor进行切片，返回一个list
        in_dim = K.int_shape(inputs)[-1]
        assert in_dim == 152
        # 忽略掉149,150,151
        return [inputs[:, :72], inputs[:, 72:72+76], inputs[:, 148+3:152]]

    def compute_output_shape(self, input_shape):
        # output_shape也要是对应的list
        # in_dim = input_shape[-1]
        # user_features,item_features,label强相关的item_feature
        return [(None, 72), (None, 76), (None, 1)]


class SplitUserItem(Layer):
    # 方便UI双塔
    def __init__(self, **kwargs):
        super(SplitUserItem, self).__init__(**kwargs)

    def call(self, inputs):
        # 按第二个维度对tensor进行切片，返回一个list
        in_dim = K.int_shape(inputs)[-1]
        assert in_dim == 152
        return [inputs[:, :72], inputs[:, 72:]]

    def compute_output_shape(self, input_shape):
        # output_shape也要是对应的list
        in_dim = input_shape[-1]
        return [(None, 72), (None, in_dim-72)]
