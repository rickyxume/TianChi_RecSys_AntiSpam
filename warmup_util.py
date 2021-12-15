import csv
import os
import time
import threading
from subprocess import Popen
from typing import Dict
from uuid import uuid1
import pandas as pd
import yaml
from notification_service.client import NotificationClient
from notification_service.base_notification import EventWatcher, BaseEvent
from kafka import KafkaProducer, KafkaAdminClient, KafkaConsumer
from kafka.admin import NewTopic
from typing import List
import sys
import getopt
import json
import numpy as np


def init_kafka(bootstrap_servers, input_topic, output_topic):
    admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_servers)
    topics = admin_client.list_topics()
    if input_topic not in topics:
        print("create input topic: "+input_topic)
        admin_client.create_topics(
            new_topics=[NewTopic(name=input_topic, num_partitions=1, replication_factor=1)])
    if output_topic not in topics:
        print("create output topic: "+output_topic)
        admin_client.create_topics(
            new_topics=[NewTopic(name=output_topic, num_partitions=1, replication_factor=1)])


def push_kafka(bootstrap_servers, input_filename, input_topic):
    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: v.encode())
    np.random.seed(1)
    # feat = np.array([np.apply_along_axis(lambda x: ' '.join(x.astype(str).tolist()), 1, np.random.rand(2, 152).repeat(2500, axis=0))]).T
    feat = np.array([np.apply_along_axis(lambda x: ' '.join(
        x.astype(str).tolist()), 1, np.tile(np.random.rand(2, 152), (10000, 1)))]).T
    index = np.array([np.arange(1, feat.shape[0] + 1)]).T.astype(str)
    fake_pred = np.concatenate([index] * 4 + [feat], axis=1).astype(str)

    for line in fake_pred:
        line = ','.join(line)
        producer.send(input_topic, value=line)
        time.sleep(0.008)
    time.sleep(10)


def listen_kafka(bootstrap_servers, output_filename, input_topic, output_topic):
    consumer = KafkaConsumer(
        input_topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest',
        consumer_timeout_ms=1000
    )
    input_time = {}
    for message in consumer:
        input_time[int(message.value.decode().split(',')[0])
                   ] = message.timestamp
    print('received ' + str(len(input_time)) + ' messages from input topic.')
    time.sleep(10)

    consumer = KafkaConsumer(
        output_topic,
        bootstrap_servers=bootstrap_servers,
        auto_offset_reset='earliest',
        consumer_timeout_ms=1000
    )
    output_time = {}
    output_label = {}
    for message in consumer:
        line = message.value.decode().strip()
        uid = int(line.split(',')[0])
        output_time[uid] = message.timestamp
        output_label[uid] = int(json.loads(','.join(line.split(',')[1:])[
                                1:-1].replace('""', '"'))['data'][0])
    print('received ' + str(len(output_time)) + ' messages from output topic.')

    resultf = open(output_filename, 'w+')
    for uid in input_time:
        if uid not in output_label or uid not in output_time:
            continue
        resultf.writelines(['{},{},{},{}\n'.format(
            uid, input_time[uid], output_time[uid], output_label[uid])])
    print('kafka messages have been written to ' + output_filename)


class KafkaWatcher(EventWatcher):
    def __init__(self, bootstrap_servers, input_topic, output_topic):
        super().__init__()
        self.bootstrap_servers = bootstrap_servers
        self.input_topic = input_topic
        self.output_topic = output_topic

    def process(self, events: List[BaseEvent]):
        print("warmup event triggered ")
        time.sleep(20)
        push_kafka(self.bootstrap_servers,
                   '/tcdata/train0.csv', self.input_topic)
        listen_kafka(self.bootstrap_servers, './result.csv',
                     self.input_topic, self.output_topic)
        sys.exit()


if __name__ == '__main__':
    opts, args = getopt.getopt(
        sys.argv[1:], "", ["input_topic=", "output_topic=", "server="])
    mydict = dict(opts)
    input_topic = mydict.get('--input_topic', '')
    output_topic = mydict.get('--output_topic', '')
    bootstrap_servers = mydict.get('--server', '')
    bootstrap_servers = bootstrap_servers.split(',')

    init_kafka(bootstrap_servers, input_topic, output_topic)

    notification_client = NotificationClient(
        'localhost:50051', default_namespace="default")
    notification_client.start_listen_event(key='KafkaWatcher', event_type='UNDEFINED', namespace="default",
                                           watcher=KafkaWatcher(
                                               bootstrap_servers, input_topic, output_topic),
                                           start_time=0)
