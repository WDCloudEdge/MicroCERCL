total_count=$1
dir="$(dirname "$0")"
is_scale=$2
# sh $dir/chaos_service.sh "details" $total_count $is_scale
# sh $dir/chaos_service.sh "productpage" $total_count $is_scale
# sh $dir/chaos_service.sh "ratings" $total_count $is_scale
# sh $dir/chaos_service.sh "reviews-v1" $total_count $is_scale
sh $dir/chaos_service.sh "reviews-v2" $total_count $is_scale
# sh $dir/chaos_service.sh "reviews-v3" $total_count $is_scale

