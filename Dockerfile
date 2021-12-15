# 复赛原始镜像
#FROM registry.cn-shanghai.aliyuncs.com/tcc_public/tianchi_antispam:v2
# 修复后的复赛镜像
FROM registry.cn-shanghai.aliyuncs.com/yoikl/test:v2_fixed
# 预置工具
# RUN apt update && apt install -y fish vim
RUN pip install h5py==2.10
# 线下模拟 小内存
# ARG env=local
# COPY /${env}/run_bash.sh          /root/tianchi_entry/run_bash.sh
# COPY /${env}/flink-conf.yaml       /opt/flink-1.11.2/conf/flink-conf.yaml

# 线上 大内存
ARG env=online
COPY /${env}/run_bash.sh           /root/tianchi_entry/run_bash.sh
COPY /${env}/flink-conf.yaml       /opt/flink-1.11.2/conf/flink-conf.yaml
COPY /${env}/Occlum.json           /root/occlum_builder/Occlum.json
COPY /${env}/run_flink_fish.sh     /root/tianchi_occlum/image/bin/run_flink_fish.sh
COPY /${env}/flink                 /root/tianchi_occlum/image/bin/flink
COPY /${env}/python                /root/tianchi_occlum/image/bin/python


COPY /tf_main.py           /root/tianchi_aiflow/workflows/tianchi_main/tf_main.py
COPY /tianchi_executor.py  /root/tianchi_aiflow/workflows/tianchi_main/tianchi_executor.py
COPY /tianchi_main.py      /root/tianchi_aiflow/workflows/tianchi_main/tianchi_main.py
COPY /workflow_util.py     /root/tianchi_entry/workflow_util.py
COPY /warmup_util.py       /root/tianchi_entry/warmup_util.py
COPY /score.py             /root/tianchi_entry/score.py

ADD Models.py             /root/tianchi_aiflow/workflows/tianchi_main
ADD utils.py              /root/tianchi_aiflow/workflows/tianchi_main

RUN chmod 777             /root/tianchi_occlum/image/bin/python /root/tianchi_occlum/image/bin/run_flink_fish.sh /root/tianchi_entry/run_bash.sh /root/tianchi_occlum/image/bin/flink
