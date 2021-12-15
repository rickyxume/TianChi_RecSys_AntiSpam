#! /usr/bin/fish
set flink_path "/opt/flink-1.11.2"
function run_taskmanager
    ulimit -Sv 30000000
    /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
        -XX:+UseG1GC -Xmx10g -Xms10g -XX:MaxDirectMemorySize=9g -XX:MaxMetaspaceSize=256m \
        -XX:ActiveProcessorCount=2 \
        -Dlog.file=/host/tm.log \
        -Dos.name=Linux \
        -Dlog4j.configurationFile=file:$flink_path/conf/log4j.properties \
        -Dlogback.configurationFile=file:$flink_path/conf/logback.xml \
        -Dorg.apache.flink.shaded.netty4.io.netty.eventLoopThreads=1 \
	-Djdk.lang.Process.launchMechanism=posix_spawn \
        -classpath $flink_path/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.10.0-serving.jar:$flink_path/lib/flink-table-blink_2.11-1.11.2.jar:$flink_path/lib/flink-table_2.11-1.11.2.jar:$flink_path/lib/flink-dist_2.11-1.11.2.jar:$flink_path/lib/log4j-1.2-api-2.12.1.jar:$flink_path/lib/log4j-slf4j-impl-2.12.1.jar:$flink_path/lib/log4j-api-2.12.1.jar:$flink_path/lib/flink-csv-1.11.2.jar:$flink_path/lib/flink-json-1.11.2.jar:$flink_path/lib/flink-shaded-zookeeper-3.4.14.jar:$flink_path/lib/log4j-core-2.12.1.jar org.apache.flink.runtime.taskexecutor.TaskManagerRunner \
        --configDir $flink_path/conf \
	-D taskmanager.memory.framework.off-heap.size=128mb \
	-D taskmanager.memory.network.max=1024mb \
	-D taskmanager.memory.network.min=1024mb \
	-D taskmanager.memory.framework.heap.size=128mb \
	-D taskmanager.memory.managed.size=12gb \
	-D taskmanager.cpu.cores=1.0 \
	-D taskmanager.memory.task.heap.size=9gb \
	-D taskmanager.memory.task.off-heap.size=8gb \
    > /host/tm.log &
end

function run_jobmanager
    ulimit -Sv 11240000
    /usr/lib/jvm/java-11-openjdk-amd64/bin/java \
	-Xmx7g -Xms7g -XX:MaxMetaspaceSize=256m \
        -XX:ActiveProcessorCount=2 \
        -Dlog.file=/host/jm.log \
        -Dos.name=Linux \
        -Dlog4j.configurationFile=file:$flink_path/conf/log4j.properties \
        -Dlogback.configurationFile=file:$flink_path/conf/logback.xml \
	-Djdk.lang.Process.launchMechanism=posix_spawn \
        -classpath $flink_path/lib/analytics-zoo-bigdl_0.12.2-spark_2.4.3-0.10.0-serving.jar:$flink_path/lib/flink-table-blink_2.11-1.11.2.jar:$flink_path/lib/flink-table_2.11-1.11.2.jar:$flink_path/lib/flink-dist_2.11-1.11.2.jar:$flink_path/lib/log4j-1.2-api-2.12.1.jar:$flink_path/lib/log4j-slf4j-impl-2.12.1.jar:$flink_path/lib/log4j-api-2.12.1.jar:$flink_path/lib/flink-csv-1.11.2.jar:$flink_path/lib/flink-json-1.11.2.jar:$flink_path/lib/flink-shaded-zookeeper-3.4.14.jar:$flink_path/lib/log4j-core-2.12.1.jar org.apache.flink.runtime.entrypoint.StandaloneSessionClusterEntrypoint \
        --configDir $flink_path/conf \
        --executionMode cluster \
        > /host/jm.log &
end

set arg $argv[1]
switch $arg
    case jm
        run_jobmanager
    case tm
        run_taskmanager
end