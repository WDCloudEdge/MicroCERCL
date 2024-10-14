#sh carts-cloud/cpu_load.sh 10 0 'carts-cloud' >> carts-cloud.txt
#sh carts-cloud/mem_load.sh 10 0 'carts-cloud' >> carts-cloud.txt
#sh carts-cloud/net_latency.sh 10 0 'carts-cloud' >> carts-cloud.txt
#
sh payment-cloud/cpu_load.sh 10 0 'payment-cloud' >> payment-cloud.txt
sh payment-cloud/mem_load.sh 10 0 'payment-cloud' >> payment-cloud.txt
sh payment-cloud/net_latency.sh 10 0 'payment-cloud' >> payment-cloud.txt

#
#sh orders-edge/cpu_load.sh 10 0 'orders-edge' >> orders-edge.txt
#sh orders-edge/mem_load.sh 10 0 'orders-edge' >> orders-edge.txt
#sh orders-edge/net_latency.sh 10 0 'orders-edge' >> orders-edge.txt

#sh shipping-cloud/cpu_load.sh 10 0 'shipping-cloud' >> shipping-cloud.txt
#sh shipping-cloud/mem_load.sh 10 0 'shipping-cloud' >> shipping-cloud.txt
#sh shipping-cloud/net_latency.sh 10 0 'shipping-cloud' >> shipping-cloud.txt


#sh frontend-cloud/cpu_load.sh 10 0 'frontend-cloud' >> frontend-cloud.txt
#sh frontend-cloud/mem_load.sh 10 0 'frontend-cloud' >> frontend-cloud.txt
#sh frontend-cloud/net_latency.sh 10 0 'frontend-cloud' >> frontend-cloud.txt
#
#sh orders-cloud/cpu_load.sh 10 0 'orders-cloud' >> orders-cloud.txt
#sh orders-cloud/mem_load.sh 10 0 'orders-cloud' >> orders-cloud.txt
#sh orders-cloud/net_latency.sh 10 0 'orders-cloud' >> orders-cloud.txt
