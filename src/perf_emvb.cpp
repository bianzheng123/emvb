#include <iostream>
#include <vector>
#include <set>
#include <chrono>
#include <cfloat>
#include <string>
#include <tuple>
#include <queue>
#include <fstream>
#include <cstdint>
#include <omp.h>
#include <cnpy.h>

#include "include/parser.hpp"

#include "DocumentScorer.hpp"

using namespace std;

void configure(cmd_line_parser::parser& parser)
{
    parser.add("k", "Number of nearest neighbours.", "-k", false);
    parser.add("nprobe", "Number of cell to look during index search.", "-nprobe", false);
    parser.add("index_dir_path", "Path to the decomposed index", "-index-dir-path", false);
    parser.add("thresh", "Threshold", "-thresh", false);
    parser.add("thresh_query", "Threshold", "-thresh-query", false);

    parser.add("out_second_stage", "Number of candidate documents selected with bitvectors", "-out-second-stage",
               false);
    parser.add("n_doc_to_score", "Number of document to score", "-n-doc-to-score", false);
    parser.add("n_thread", "Number of document to score", "-n-thread", false);
    parser.add("queries_id_file", "Path to queries_id file", "-queries-id-file", false);
    // todo remove in the future questo troiaio (id only tsv)
    parser.add("alldoclens_path", "Path to the doclens file", "-alldoclens-path", false);
    parser.add("outputfile", "Path to the output file used to compute the metrics", "-out-file", false);
}

int main(int argc, char** argv)
{
    omp_set_num_threads(1);

    cmd_line_parser::parser parser(argc, argv);
    configure(parser);
    bool success = parser.parse();
    if (!success)
        return 1;

    int k = parser.get<int>("k");
    float thresh = parser.get<float>("thresh");
    float thresh_query = parser.get<float>("thresh_query");


    size_t n_doc_to_score = parser.get<size_t>("n_doc_to_score");
    size_t n_thread = parser.get<size_t>("n_thread");
    size_t nprobe = parser.get<size_t>("nprobe");
    size_t out_second_stage = parser.get<size_t>("out_second_stage");
    string queries_id_file = parser.get<string>("queries_id_file");
    string index_dir_path = parser.get<string>("index_dir_path");
    string alldoclens_path = parser.get<string>("alldoclens_path");
    string outputfile = parser.get<string>("outputfile");

    string queries_path = index_dir_path + "/query_embeddings.npy";

    cnpy::NpyArray queriesArray = cnpy::npy_load(queries_path);

    size_t n_queries = queriesArray.shape[0];
    size_t vec_per_query = queriesArray.shape[1];
    size_t len = queriesArray.shape[2];

    cout << "Dimension: " << len << "\n"
        << "Number of queries: " << n_queries << "\n"
        << "Vector per query " << vec_per_query << "\n";
    uint16_t values_per_query = vec_per_query * len;
    valType* loaded_query_data = queriesArray.data<valType>();

    // load qid mapping file
    auto qid_map = load_qids(queries_id_file);

    cout << "queries id loaded\n";

    // load documents
    const string doclens_path = alldoclens_path;
    const string decomposed_index_path = index_dir_path;
    const size_t max_query_terms = vec_per_query;

    string codes_path = decomposed_index_path + "/residuals.npy";
    const NpyArray pqCodesArray = cnpy::npy_load(codes_path);

    string centroids_path = decomposed_index_path + "/centroids.npy";
    const NpyArray centroidsArray = cnpy::npy_load(centroids_path);

    string centroids_assignment_path = decomposed_index_path + "/index_assignment.npy";
    const NpyArray centroidsAssignmentArray = cnpy::npy_load(centroids_assignment_path);

    const NpyArray doclensArray = cnpy::npy_load(doclens_path);

    string pq_centroids_path = decomposed_index_path + "/pq_centroids.npy";
    const NpyArray pqCentroidsArray = cnpy::npy_load(pq_centroids_path);

    std::vector<std::unique_ptr<DocumentScorer>> document_scorer_l(n_thread);

    for(int thread_id = 0; thread_id < n_thread; thread_id++)
    {
        document_scorer_l[thread_id] = std::make_unique<DocumentScorer>(doclensArray, centroidsArray, centroidsAssignmentArray,
                                   pqCodesArray, pqCentroidsArray,
                                   decomposed_index_path, max_query_terms);
    }

    // DocumentScorer document_scorer(doclensArray, centroidsArray, centroidsAssignmentArray,
                                   // pqCodesArray, pqCentroidsArray,
                                   // decomposed_index_path, max_query_terms);

    // uint64_t tot_time_score = 0;
    // uint64_t time_centroids_selection = 0;
    // uint64_t time_second_stage = 0;

    // uint64_t time_document_filtering = 0;

    std::vector<std::vector<std::pair<unsigned long, float>>> result_topk_l(n_queries);
    for(uint32_t query_id=0;query_id < n_queries;query_id++)
    {
        result_topk_l[query_id].resize(k);
    }
    auto start = chrono::high_resolution_clock::now();
    cout << "SEARCH STARTED\n";
#pragma omp parallel for default(none) shared(n_queries, values_per_query, document_scorer_l, thresh, nprobe, loaded_query_data, n_doc_to_score, out_second_stage, thresh_query, k) num_threads(n_thread)
    for (size_t query_id = 0; query_id < n_queries; query_id++)
    {
        const int threadID = omp_get_thread_num();
        globalIdxType q_start = query_id * values_per_query;

        // PHASE 1: candidate documents retrieval
        auto candidate_docs =  document_scorer_l[threadID]->find_candidate_docs(loaded_query_data, q_start, nprobe, thresh);


        // PHASE 2: candidate document filtering
        auto selected_docs =  document_scorer_l[threadID]->compute_hit_frequency(candidate_docs, thresh, n_doc_to_score);

        //  PHASE 3: second stage filtering
        auto selected_docs_2nd =  document_scorer_l[threadID]->second_stage_filtering(loaded_query_data, q_start, selected_docs,
                                                                        out_second_stage);

        // PHASE 4: document scoring
        auto query_res =  document_scorer_l[threadID]->compute_topk_documents_selected(
            loaded_query_data, q_start, selected_docs_2nd, k, thresh_query);
    }
    uint64_t total_time = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - start).count();

    ofstream out_file; // file with final output
    out_file.open(outputfile);
    for(size_t query_id = 0; query_id < n_queries; query_id++)
    {
        for(int i = 0; i < k; i++)
        {
            out_file << qid_map[query_id] << "\t" << result_topk_l[query_id][i].first << "\t" << i + 1 << "\t" << result_topk_l[query_id][i].second << endl;
        }
    }

    out_file.flush();
    out_file.close();
    cout << "Average Elapsed Time per query " << total_time / n_queries << "\n";

    return 0;
}
