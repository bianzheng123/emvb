index_path="/home/bianzheng/Dataset/billion-scale-multi-vector-retrieval/Index/lotte-500-gnd/emvb"
alldoclens_path="/home/bianzheng/Dataset/billion-scale-multi-vector-retrieval/Index/lotte-500-gnd/emvb/doclens.npy"
queries_id_file="/home/bianzheng/Dataset/billion-scale-multi-vector-retrieval/Index/lotte-500-gnd/emvb/qID_l.txt"
./build/perf_emvb -k 10 -nprobe 4 -thresh 0.4 -out-second-stage 50 -thresh-query 0.5 -n-doc-to-score 100 -n-thread 1 -queries-id-file $queries_id_file  -alldoclens-path $alldoclens_path -index-dir-path $index_path -out-file results_10_lotte.tsv
#./build/perf_emvb -k 10 -nprobe 4 -thresh 0.4 -out-second-stage 50 -thresh-query 0.5 -n-doc-to-score 100 -queries-id-file $queries_id_file  -alldoclens-path $alldoclens_path -index-dir-path $index_path -out-file results_10_lotte.tsv
