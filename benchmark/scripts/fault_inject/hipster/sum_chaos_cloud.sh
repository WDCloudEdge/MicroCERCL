#sh recommendationservice-edge/cpu_load.sh 5 0 'recommendationservice-edge' >> recommendationservice-edge.txt
#sh recommendationservice-edge/mem_load.sh 5 0 'recommendationservice-edge' >> recommendationservice-edge.txt
#sh recommendationservice-edge/net_latency.sh 5 0 'recommendationservice-edge' >> recommendationservice-edge.txt
#
sh recommendationservice-edge/cpu_load.sh 5 0 'recommendationservice-edge' >> recommendationservice-edge.txt
sh recommendationservice-edge/mem_load.sh 5 0 'recommendationservice-edge' >> recommendationservice-edge.txt
sh recommendationservice-edge/net_latency.sh 5 0 'recommendationservice-edge' >> recommendationservice-edge.txt

sh cartservice/cpu_load.sh 5 0 'cartservice' >> cartservice.txt
sh cartservice/mem_load.sh 5 0 'cartservice' >> cartservice.txt
sh cartservice/net_latency.sh 5 0 'cartservice' >> cartservice.txt
##
#sh emailservice/cpu_load.sh 5 0 'emailservice' >> emailservice.txt
#sh emailservice/mem_load.sh 5 0 'emailservice' >> emailservice.txt
#sh emailservice/net_latency.sh 5 0 'emailservice' >> emailservice.txt

