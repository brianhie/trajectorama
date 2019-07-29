rm -rf monitor_memory.log
while ps -p $1 > /dev/null
do
    date >> monitor_memory.log
    free -m >> monitor_memory.log
    sleep 0.5
done
