#ifndef WORD2VEC_H
#define WORD2VEC_H

#include <vector>
#include <list>
#include <string>
#include <set>
#include <unordered_map>
#include <tuple>
#include <algorithm>
#include <numeric>
#include <random>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iterator>
#include <cstdint>
#include <Eigen/Dense>

#include "Word.h"

using namespace std;
using namespace Eigen;

#define EIGEN_NO_DEBUG

typedef Matrix<float, Dynamic, Dynamic, RowMajor> RMatrixXf;

class Word2Vec
{
public:
	int iter;
	int window;
	int min_count;
	int table_size;
	int word_dim;
	int negative;                  //num of negetive samples
	float subsample_threshold;
	float init_alpha;
	float min_alpha;
	
	bool cbow_mean;
	string train_method;
	string model;

	vector<Word *> vocab;
	vector<string> idx2word;
	unordered_map<string, WordP> vocab_hash;
	vector<size_t> table;

	RMatrixXf W, synapses1, synapses1_neg;

	std::random_device rd;
	std::mt19937 generator;
	std::uniform_int_distribution<int> distribution_window;
	std::uniform_int_distribution<int> distribution_table;


public:
	~Word2Vec(void);

	Word2Vec(int iter=1, int window=5, int min_count=5, int table_size=100000000, int word_dim=200, 
		int negative=0, float subsample_threshold=0.001, float init_alpha=0.025, float min_alpha=1e-6,
		bool cbow_mean=false, string train_method="hs", string model="cbow");

	vector<vector<string>> line_docs(string file_name);
	void reduce_vocab();
	void create_huffman_tree();
	void make_table();
	void precalc_sampling();
	void build_vocab(vector<vector<string>> &sentences);
	void save_vocab(string vocab_filename);
	void read_vocab(string vocab_filename);

	void init_weights(size_t vocab_size);

	vector<vector<Word *>> build_sample(vector<vector<string>> & data);

	RowVectorXf& hierarchical_softmax(Word * predict_word, RowVectorXf& project_rep, RowVectorXf& project_grad, float alpha);
	RowVectorXf& negative_sampling(Word * predict_word, RowVectorXf& project_rep, RowVectorXf& project_grad, float alpha);
	void train_sentence_cbow(vector<Word *>& sentence, float alpha);
	void train_sentence_sg(vector<Word *>& sentence, float alpha);

	void train(vector<vector<string>> &sentences);

	void save_word2vec(string filename, bool binary=false);
	void load_word2vec(string word2vec_filename, bool binary);
};

#endif