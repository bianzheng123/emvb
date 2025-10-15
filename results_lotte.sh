
qrels_file="aux_data/lotte/qrels.dev.tsv"
index_path="/home/zhengbian/Dataset/vector-set-similarity-search/Index/lotte/260k_m32_LOTTE_OPQ"


# k = 10
export OMP_NUM_THREADS=1
./build/perf_emvb -k 10 -nprobe 4 -thresh 0.4 -out-second-stage 128 -thresh-query 0.4 -n-doc-to-score 200 -queries-id-file aux_data/lotte/queries_id_lotte.tsv  -alldoclens-path aux_data/lotte/doclens_lotte.npy -index-dir-path $index_path -out-file results_10_lotte.tsv
export OMP_NUM_THREADS=8
./build/perf_emvb -k 10 -nprobe 4 -thresh 0.4 -out-second-stage 128 -thresh-query 0.4 -n-doc-to-score 200 -queries-id-file aux_data/lotte/queries_id_lotte.tsv  -alldoclens-path aux_data/lotte/doclens_lotte.npy -index-dir-path $index_path -out-file results_10_lotte.tsv
export OMP_NUM_THREADS=48
./build/perf_emvb -k 10 -nprobe 4 -thresh 0.4 -out-second-stage 128 -thresh-query 0.4 -n-doc-to-score 200 -queries-id-file aux_data/lotte/queries_id_lotte.tsv  -alldoclens-path aux_data/lotte/doclens_lotte.npy -index-dir-path $index_path -out-file results_10_lotte.tsv
#
#python evaluate_lotte_rankings.py -gt aux_data/lotte/lotte_pooled_qas.search.jsonl -r results_10_lotte.tsv
python compute_mrr.py --qrels $qrels_file --ranking results_10_lotte.tsv

# k = 100
export OMP_NUM_THREADS=1
./build/perf_emvb -k 100 -nprobe 4 -thresh 0.4 -out-second-stage 128 -thresh-query 0.4 -n-doc-to-score 200 -queries-id-file aux_data/lotte/queries_id_lotte.tsv  -alldoclens-path aux_data/lotte/doclens_lotte.npy -index-dir-path $index_path -out-file results_100_lotte.tsv
export OMP_NUM_THREADS=8
./build/perf_emvb -k 100 -nprobe 4 -thresh 0.4 -out-second-stage 128 -thresh-query 0.4 -n-doc-to-score 200 -queries-id-file aux_data/lotte/queries_id_lotte.tsv  -alldoclens-path aux_data/lotte/doclens_lotte.npy -index-dir-path $index_path -out-file results_100_lotte.tsv
export OMP_NUM_THREADS=48
./build/perf_emvb -k 100 -nprobe 4 -thresh 0.4 -out-second-stage 128 -thresh-query 0.4 -n-doc-to-score 200 -queries-id-file aux_data/lotte/queries_id_lotte.tsv  -alldoclens-path aux_data/lotte/doclens_lotte.npy -index-dir-path $index_path -out-file results_100_lotte.tsv
#
#python evaluate_lotte_rankings.py -gt aux_data/lotte/lotte_pooled_qas.search.jsonl -r results_100_lotte.tsv
python compute_mrr.py --qrels $qrels_file --ranking results_100_lotte.tsv


qrels_file="aux_data/msmarco/qrels.dev.tsv"
index_path_1="/home/zhengbian/Dataset/multi-vector-retrieval/Index/msmacro/260_m16"

export OMP_NUM_THREADS=1
./build/perf_emvb -k 10 -nprobe 4 -thresh 0.4 -out-second-stage 128 -thresh-query 0.4 -n-doc-to-score 200 -queries-id-file aux_data/msmarco/queries_dev_small_idonly.tsv  -alldoclens-path aux_data/msmarco/doclens_msmarco.npy -index-dir-path $index_path_1 -out-file results_10.tsv
export OMP_NUM_THREADS=8
./build/perf_emvb -k 10 -nprobe 4 -thresh 0.4 -out-second-stage 128 -thresh-query 0.4 -n-doc-to-score 200 -queries-id-file aux_data/msmarco/queries_dev_small_idonly.tsv  -alldoclens-path aux_data/msmarco/doclens_msmarco.npy -index-dir-path $index_path_1 -out-file results_10.tsv
export OMP_NUM_THREADS=48
./build/perf_emvb -k 10 -nprobe 4 -thresh 0.4 -out-second-stage 128 -thresh-query 0.4 -n-doc-to-score 200 -queries-id-file aux_data/msmarco/queries_dev_small_idonly.tsv  -alldoclens-path aux_data/msmarco/doclens_msmarco.npy -index-dir-path $index_path_1 -out-file results_10.tsv
python compute_mrr.py --qrels $qrels_file --ranking results_10.tsv


export OMP_NUM_THREADS=1
./build/perf_emvb -k 100 -nprobe 4 -thresh 0.4 -out-second-stage 128 -thresh-query 0.4 -n-doc-to-score 200 -queries-id-file aux_data/msmarco/queries_dev_small_idonly.tsv  -alldoclens-path aux_data/msmarco/doclens_msmarco.npy -index-dir-path $index_path_1 -out-file results_100.tsv
export OMP_NUM_THREADS=8
./build/perf_emvb -k 100 -nprobe 4 -thresh 0.4 -out-second-stage 128 -thresh-query 0.4 -n-doc-to-score 200 -queries-id-file aux_data/msmarco/queries_dev_small_idonly.tsv  -alldoclens-path aux_data/msmarco/doclens_msmarco.npy -index-dir-path $index_path_1 -out-file results_100.tsv
export OMP_NUM_THREADS=48
./build/perf_emvb -k 100 -nprobe 4 -thresh 0.4 -out-second-stage 128 -thresh-query 0.4 -n-doc-to-score 200 -queries-id-file aux_data/msmarco/queries_dev_small_idonly.tsv  -alldoclens-path aux_data/msmarco/doclens_msmarco.npy -index-dir-path $index_path_1 -out-file results_100.tsv
python compute_mrr.py --qrels $qrels_file --ranking results_100.tsv
