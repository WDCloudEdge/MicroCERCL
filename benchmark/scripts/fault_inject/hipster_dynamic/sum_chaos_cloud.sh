
sh adservice/cpu_load.sh 10 0 'adservice' >> adservice.txt
sh adservice/mem_load.sh 10 0 'adservice' >> adservice.txt
sh adservice/net_latency.sh 10 0 'adservice' >> adservice.txt

sh cartservice/cpu_load.sh 10 0 'cartservice' >> cartservice.txt
sh cartservice/mem_load.sh 10 0 'cartservice' >> cartservice.txt
sh cartservice/net_latency.sh 10 0 'cartservice' >> cartservice.txt
##
#sh emailservice/cpu_load.sh 5 0 'emailservice' >> emailservice.txt
#sh emailservice/mem_load.sh 5 0 'emailservice' >> emailservice.txt
#sh emailservice/net_latency.sh 5 0 'emailservice' >> emailservice.txt

