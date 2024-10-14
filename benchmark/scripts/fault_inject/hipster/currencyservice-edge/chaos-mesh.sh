# sh /home/cron/jupiter-hybrid/bookinfo/details/cpu_load.sh $1>> details-result.txt
# sh /home/cron/jupiter-hybrid/bookinfo/details/mem_load.sh $1>> details-result.txt
# sh /home/cron/jupiter-hybrid/bookinfo/details/net_latency.sh $1>> details-result.txt
dir=$PWD
sh $dir/cpu_load.sh $1>> label.txt
sh $dir/mem_load.sh $1>> label.txt
sh $dir/net_latency.sh $1>> label.txt