import time
import os
from typing import List

import ai_flow as af
from ai_flow_plugins.job_plugins import python, flink
from pyflink.table import Table
from tf_main import train
from notification_service.client import NotificationClient
from notification_service.base_notification import BaseEvent
from ai_flow.model_center.entity.model_version_stage import ModelVersionStage


def get_dependencies_path():
    return "/opt"


class TrainModel(python.PythonProcessor):
    def __init__(self) -> None:
        super().__init__()

    def first_time(self):
        # 修正路径
        return not os.path.exists('/host/model/frozen_model/frozen_inference_graph.pb')

    def process(self, execution_context: python.python_processor.ExecutionContext, input_list: List) -> List:
        print('train_job triggered ')

        model_path = '/host/model'
        save_name = 'saved_model'

        if self.first_time():
            train_path = '/tcdata/train0.csv'
            train(train_path, model_path, save_name)

            model_meta = execution_context.config['model_info']
            af.register_model_version(model=model_meta, model_path=model_path)
            model_version_meta = af.get_latest_generated_model_version(
                model_meta.name)
            deployed_model_version = af.get_deployed_model_version(
                model_name=model_meta.name)
            if deployed_model_version is not None:
                af.update_model_version(model_name=model_meta.name,
                                        model_version=deployed_model_version.version,
                                        current_stage=ModelVersionStage.DEPRECATED)
            af.update_model_version(model_name=model_meta.name,
                                    model_version=model_version_meta.version,
                                    current_stage=ModelVersionStage.VALIDATED)
            af.update_model_version(model_name=model_meta.name,
                                    model_version=model_version_meta.version,
                                    current_stage=ModelVersionStage.DEPLOYED)

        else:
            train_path = '/tcdata/train1.csv'
            train(train_path, model_path, save_name)

            model_meta = execution_context.config['model_info']
            af.register_model_version(model=model_meta, model_path=model_path)
            model_version_meta = af.get_latest_generated_model_version(
                model_meta.name)
            deployed_model_version = af.get_deployed_model_version(
                model_name=model_meta.name)
            if deployed_model_version is not None:
                af.update_model_version(model_name=model_meta.name,
                                        model_version=deployed_model_version.version,
                                        current_stage=ModelVersionStage.DEPRECATED)
            af.update_model_version(model_name=model_meta.name,
                                    model_version=model_version_meta.version,
                                    current_stage=ModelVersionStage.VALIDATED)
            af.update_model_version(model_name=model_meta.name,
                                    model_version=model_version_meta.version,
                                    current_stage=ModelVersionStage.DEPLOYED)

        return []


class Source(flink.flink_processor.FlinkSqlProcessor):
    def open(self, execution_context: flink.ExecutionContext):
        t_env = execution_context.table_env
        t_env.get_config().set_python_executable('/opt/python-occlum/bin/python3.7')
        t_env.get_config().get_configuration().set_boolean(
            "python.fn-execution.memory.managed", True)
        t_env.get_config().get_configuration().set_string(
            "classloader.resolve-order", "parent-first")
        t_env.get_config().get_configuration().set_integer(
            "python.fn-execution.bundle.size", 1)

    def sql_statements(self, execution_context: flink.ExecutionContext) -> List[str]:
        data_meta = execution_context.config['dataset']

        sql_statements = '''
            CREATE TABLE input_table (
                uuid STRING,
                visit_time STRING,
                user_id STRING,
                item_id STRING,
                features STRING
            ) WITH (
                'connector' = 'kafka',
                'topic' = '{}',
                'properties.bootstrap.servers' = '{}',
                'properties.group.id' = 'testGroup',
                'format' = '{}'
            )
        '''.format(data_meta.name, data_meta.uri, data_meta.data_format)
        return [sql_statements]


class Transformer(flink.flink_processor.FlinkSqlProcessor):
    def open(self, execution_context: flink.ExecutionContext):
        t_env = execution_context.table_env
        model_name = execution_context.config['model_info'].name
        model_version_meta = af.get_deployed_model_version(model_name)
        model_path = model_version_meta.model_path

        t_env.get_config().get_configuration().set_string('pipeline.global-job-parameters',
                                                          '"modelPath:""{}"""'
                                                          .format(os.path.join(model_path, 'frozen_model')))
        t_env.get_config().get_configuration().set_string("pipeline.classpaths",
                                                          "file://{}/flink-sql-connector-kafka_2.11-1.11.2.jar"
                                                          .format(get_dependencies_path()))

    def udf_list(self, execution_context: flink.ExecutionContext) -> List:
        t_env = execution_context.table_env
        # t_env.get_config().get_configuration().set_integer("parallelism.default", 2)
        udf_func = flink.flink_processor.UDFWrapper("cluster_serving",
                                                    "com.intel.analytics.zoo.serving.operator.ClusterServingFunction")
        return [udf_func]

    def sql_statements(self, execution_context: flink.ExecutionContext) -> List[str]:
        # 在这之前要把features处理好，可以在input_table上执行窗口函数获得历史统计特征
        preprocess_stmt = """
        """

        process_stmt = """
        CREATE VIEW processed_table AS
            SELECT uuid, cluster_serving(uuid, all_features) AS data
            FROM (SELECT uuid,CONCAT_WS(' ',it.user_id,it.item_id,it.features) as all_features
                FROM input_table it)
        """
        # original_process_stmt = "CREATE VIEW processed_table AS SELECT uuid, cluster_serving(uuid, features) AS data FROM input_table"
        # process_stmt = 'CREATE VIEW processed_table AS SELECT uuid, cluster_serving(uuid, features) AS data FROM input_table'
        return [process_stmt]


class Sink(flink.flink_processor.FlinkSqlProcessor):
    def sql_statements(self, execution_context: flink.ExecutionContext) -> List[str]:
        data_meta = execution_context.config['dataset']
        create_stmt = '''
            CREATE TABLE write_table (
                uuid STRING,
                data STRING
            ) WITH (
                'connector.type' = 'kafka',
                'connector.version' = 'universal',
                'connector.topic' = '{}',
                'connector.properties.zookeeper.connect' = '127.0.0.1:2181',
                'connector.properties.bootstrap.servers' = '{}',
                'connector.properties.group.id' = 'testGroup',
                'connector.properties.batch.size' = '1',
                'connector.properties.linger.ms' = '1',
                'format.type' = '{}'
            )
        '''.format(data_meta.name, data_meta.uri, data_meta.data_format)

        sink_stmt = 'INSERT INTO write_table SELECT * FROM processed_table'

        notification_client = NotificationClient(
            '127.0.0.1:50051', default_namespace="default")
        notification_client.send_event(
            BaseEvent(key='KafkaWatcher', value='model_registered'))

        return [create_stmt, sink_stmt]


"""下面是旧版本"""

# from ai_flow_plugins.job_plugins.python import PythonProcessor
# from notification_service.base_notification import EventWatcher
# import pandas as pd
# import numpy as np
# import time
# from typing import List

# import ai_flow as af
# from ai_flow_plugins.job_plugins import python, flink
# from pyflink.table import Table, ScalarFunction, DataTypes
# from pyflink.table.udf import udf
# from kafka import KafkaProducer, KafkaAdminClient, KafkaConsumer
# from kafka.admin import NewTopic
# from tf_main import train
# from subprocess import Popen
# import json
# import sys
# import getopt
# from notification_service.client import NotificationClient
# from notification_service.base_notification import EventWatcher, BaseEvent


# def get_model_path():
#     return '/host'


# def get_data_path():
#     return '/tcdata'


# def get_dependencies_path():
#     return "/opt"

# # https://ci.apache.org/projects/flink/flink-docs-release-1.13/zh/docs/dev/python/table_api_tutorial/


# class TrainModel(python.PythonProcessor):
#     def process(self, execution_context: python.python_processor.ExecutionContext, input_list: List) -> List:
#         train_path = get_data_path() + '/train.csv'
#         model_dir = get_model_path() + '/model/base_model'
#         save_name = 'base_model'
#         print("进入训练阶段")
#         train(train_path, model_dir, save_name)
#         af.register_model_version(
#             model=execution_context.config['model_info'], model_path=model_dir)
#         print(f"已注册一个模型版本，位于:{model_dir}")

#         return []


# class Source(flink.FlinkPythonProcessor):
#     def __init__(self, input_topic, output_topic) -> None:
#         super().__init__()
#         self.input_topic = input_topic
#         self.output_topic = output_topic

#     def process(self, execution_context: flink.ExecutionContext, input_list: List[Table] = None) -> List[Table]:
#         print("### {} setup done2 for {}".format(
#             self.__class__.__name__, "sads"))
#         t_env = execution_context.table_env

#         t_env.get_config().set_python_executable('/opt/python-occlum/bin/python3.7')
#         print("Source(flink.FlinkPythonProcessor)")
#         print(t_env.get_config().get_configuration().to_dict())

#         # 加上这条限制buffer等待的时间，减少数据从产生到真正被消费的延迟
#         # t_env.get_config().get_configuration().set_integer(
#         #     "execution.checkpointing.interval", 500)

#         t_env.get_config().get_configuration().set_boolean(
#             "python.fn-execution.memory.managed", True)
#         t_env.get_config().get_configuration().set_string('pipeline.global-job-parameters',
#                                                           '"modelPath:""{}/model/base_model/frozen_model"""'
#                                                           .format(get_model_path()))
#         t_env.get_config().get_configuration().set_string("pipeline.classpaths",
#                                                           "file://{}/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.10.0-serving.jar;file://{}/flink-sql-connector-kafka_2.11-1.11.2.jar"
#                                                           .format(get_dependencies_path(), get_dependencies_path()))
#         t_env.get_config().get_configuration().set_string(
#             "classloader.resolve-order", "parent-first")
#         t_env.get_config().get_configuration().set_integer(
#             "python.fn-execution.bundle.size", 1)
#         # t_env.get_config().get_configuration().set_integer(
#         #     "python.fn-execution.bundle.time", 300)
#         # t_env.get_config().get_configuration().set_string('parallelism.default', '6')
#         t_env.get_config().get_configuration().set_integer(
#             "execution.buffer-timeout", 100)

#         print("注册cluster_serving函数")
#         t_env.register_java_function("cluster_serving",
#                                      "com.intel.analytics.zoo.serving.operator.ClusterServingFunction")

#         create_input_table_sql = f'''
#             CREATE TABLE input_table (
#                 uuid STRING,
#                 visit_time STRING,
#                 user_id STRING,
#                 item_id STRING,
#                 features STRING
#             ) WITH (
#                 'connector' = 'kafka',
#                 'topic' = '{self.input_topic}',
#                 'properties.bootstrap.servers' = '127.0.0.1:9092',
#                 'properties.group.id' = 'testGroup',
#                 'format' = 'csv',
#                 'scan.startup.mode' = 'earliest-offset'
#             )
#         '''
#         print(f"建表 input_table:{create_input_table_sql}")
#         t_env.execute_sql(create_input_table_sql)

#         create_write_example_sql = f'''
#             CREATE TABLE write_example (
#                 uuid STRING,
#                 data STRING
#             ) WITH (
#                 'connector.type' = 'kafka',
#                 'connector.version' = 'universal',
#                 'connector.topic' = '{self.output_topic}',
#                 'connector.properties.zookeeper.connect' = '127.0.0.1:2181',
#                 'connector.properties.bootstrap.servers' = '127.0.0.1:9092',
#                 'connector.properties.group.id' = 'testGroup',
#                 'connector.properties.batch.size' = '1',
#                 'connector.properties.linger.ms' = '1',
#                 'format.type' = 'csv'
#             )
#         '''
#         print(f"建表 write_example:{create_write_example_sql}")
#         t_env.execute_sql(create_write_example_sql)

#         print("读数据 from_path:input_table")
#         input_table = t_env.from_path('input_table')
#         input_table.print_schema()
#         print("返回[input_table]")
#         return [input_table]


# class Predictor(flink.FlinkPythonProcessor):
#     def __init__(self):
#         super().__init__()
#         self.model_name = None

#     def setup(self, execution_context: flink.ExecutionContext):
#         self.model_name = execution_context.config['model_info']

#     def process(self, execution_context: flink.ExecutionContext, input_list: List[Table] = None) -> List[Table]:
#         result_table = input_list[0].select(
#             'uuid, cluster_serving(uuid, features)')

#         return [result_table]


# class Transformer(flink.FlinkPythonProcessor):
#     def __init__(self):
#         super().__init__()
#         self.model_name = None

#     def setup(self, execution_context: flink.ExecutionContext):
#         self.model_name = execution_context.config['model_info']

#     def process(self, execution_context: flink.ExecutionContext, input_list: List[Table] = None) -> List[Table]:
#         # 这个input_list[0]其实就是input_table
#         # t_env = execution_context.table_env
#         # t_env.get_config().set_python_executable('/opt/python-occlum/bin/python3.7')
#         print("Transformer(flink.FlinkPythonProcessor)")
#         # print(t_env.get_config().get_configuration().to_dict())
#         # t_env.get_config().get_configuration().set_integer(
#         #     "execution.checkpointing.interval", 250)
#         #
#         # user_id2user_visit_times_dict = {
#         #     "user_id": user_visit_times
#         # }

#         # input_table = input_list[0]

#         # df_input = input_table.to_pandas()
#         # df_input['user_visit_times'] = df_input['user_id'].map(
#         #     user_id2user_visit_times_dict)

#         # df_input[''] = df_input.map(user_id_dict)
#         # df_input[''] = df_input.map(user_id_dict)
#         # df_input[''] = df_input.map(user_id_dict)
#         # df_input[''] = df_input.map(user_id_dict)
#         # df_input[''] = df_input.map(user_id_dict)

#         # # 合并加到features
#         # features = " ".join([for i in df_input.values])
#         # df_input['features'] = features

#         # table = t_env.from_pandas(df_input, ['f0', 'f1'])

#         # pf转回table

#         result_table = input_list[0].select(
#             'uuid, cluster_serving(uuid, features)')

#         return [result_table]


# class Sink(flink.FlinkPythonProcessor):
#     def process(self, execution_context: flink.ExecutionContext, input_list: List[Table] = None) -> List[Table]:
#         print("### {} setup done".format(self.__class__.__name__))
#         execution_context.statement_set.add_insert(
#             "write_example", input_list[0])
#         notification_client = NotificationClient(
#             '127.0.0.1:50051', default_namespace="default")
#         notification_client.send_event(
#             BaseEvent(key='KafkaWatcher', value='model_registered'))
#         return []
# # class PredictWatcher(EventWatcher):

# #     def __init__(self):
# #         super().__init__()
# #         self.model_version = None

# #     def process(self, notifications):
# #         for notification in notifications:
# #             self.model_version = notification.value


# # class ModelPredictor(PythonProcessor):
# #     def __init__(self):
# #         super().__init__()
# #         self.model_name = None
# #         self.model_version = None
# #         self.watcher = PredictWatcher()

# #     def open(self, execution_context: ExecutionContext):
# #         # In this class, we show the usage of start_listen_event method which make it possible to send various events.
# #         # Users can also refer `stream train stream predict` dataset to directly use provided API to get model version.
# #         af.get_ai_flow_client().start_listen_event(
# #             key='START_PREDICTION', watcher=self.watcher)
# #         model_meta: af.ModelMeta = execution_context.config.get('model_info')
# #         self.model_name = model_meta.name
# #         print("### {} setup done for {}".format(
# #             self.__class__.__name__, self.model_name))

# #     def process(self, execution_context: ExecutionContext, input_list: List) -> List:
# #         while self.watcher.model_version is None:
# #             time.sleep(2)
# #         print("### {} ".format(self.watcher.model_version))

# #         def predict(df):
# #             x_test = df
# #             model_meta = af.get_deployed_model_version(self.model_name)
# #             model_path = model_meta.model_path
# #             clf = load(model_path)
# #             return model_meta.version, clf.predict(x_test)

# #         return [input_list[0].map(predict)]
