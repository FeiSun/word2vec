// require eigen, c++11, openmp(for multithread)
// g++ main.cpp Word2Vec.cpp -oword2vec -I/usr/local/include/eigen/ -std=c++11 -Ofast -march=native -funroll-loops -fopenmp
#include "Word2Vec.h"

void help()
{

	cout << "WORD VECTOR estimation toolkit v 0.1c" << endl << endl;
	cout << "Options:" << endl;
	cout << "Parameters for training:" << endl;
	cout << "\t-train <file>" << endl;
	cout << "\t\tUse text data from <file> to train the model" << endl;
	cout << "\t-output <file>" << endl;
	cout << "\t\tUse <file> to save the resulting word vectors" << endl;
	cout << "\t-size <int>"<< endl;
	cout << "\t\tSet size of word vectors; default is 200"<< endl;
	cout << "\t-window <int>"<< endl;
	cout << "\t\tSet max skip length between words; default is 5"<< endl;
	cout << "\t-subsample <float>"<< endl;
	cout << "\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data"<< endl;
	cout << "\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)"<< endl;
	cout << "\t-train_method <string>"<< endl;
	cout << "\t\tThe train_method: default is Hierarchical Softmax(hs), (ns for negative sampling)"<< endl;
	cout << "\t-negative <int>" << endl;
	cout << "\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)" << endl;
	cout << "\t-threads <int>" << endl;
	cout << "\t\tUse <int> threads (default 12)" << endl;
	cout << "\t-iter <int>" << endl;
	cout << "\t\tRun more training iterations (default 5)" << endl;
	cout << "\t-min-count <int>" << endl;
	cout << "\t\tThis will discard words that appear less than <int> times; default is 5" << endl;
	cout << "\t-alpha <float>" << endl;
	cout << "\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW" << endl;
	cout << "\t-classes <int>" << endl;
	cout << "\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)" << endl;
	cout << "\t-debug <int>" << endl;
	cout << "\t\tSet the debug mode (default = 2 = more info during training)" << endl;
	cout << "\t-binary <int>" << endl;
	cout << "\t\tSave the resulting vectors in binary moded; default is 0 (off)" << endl;
	cout << "\t-save-vocab <file>" << endl;
	cout << "\t\tThe vocabulary will be saved to <file>" << endl;
	cout << "\t-read-vocab <file>" << endl;
	cout << "\t\tThe vocabulary will be read from <file>, not constructed from the training data" << endl;
	cout << "\t-model <string>" << endl;
	cout << "\t\tThe model; default is continuous bag of words model(cbow) (use sg for skip-gram model)" << endl;
	cout << "\nExamples:" << endl;
	cout << "./word2vec -train data.txt -output vec.txt -size 200 -window 5 -subsample 1e-4 -negative 5 -model sg -train_method ns -binary 0 -iter 3" << endl;
}

int ArgPos(char *str, int argc, char **argv)
{
	for (int i = 1; i < argc; ++i)
		if (!strcmp(str, argv[i])) {
			if (i == argc - 1) {
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return i;
		}
		return -1;
}

vector<vector<string>> text8_corpus()
{
	size_t count = 0;
	const size_t max_sentence_len = 1000;
	vector<vector<string>> sentences;
	ifstream in("text8");
	vector<string> sentence;
	while(true)
	{
		string s;

		in >> s;
		if (s.empty()) break;

		count++;
		sentence.push_back(s);

		if(count == max_sentence_len)
		{
			count = 0;
			sentences.push_back(sentence);
			sentence.clear();
		}
	}
	in.close();
	if(!sentence.empty())
		sentences.push_back(sentence);

	return std::move(sentences);
}

int main(int argc, char* argv[])
{
	Eigen::initParallel();

	int i = 0;
	if (argc == 1)
	{
		help();
		return 0;
	}

	string input_file = "";
	string output_file = "text8-sgns.txt";
	string save_vocab_file = "";
	string read_vocab_file = "";
	string model = "sg";
	string train_method = "ns";
	int table_size = 100000000;
	int word_dim = 200;
	float init_alpha = 0.025f;
	int window = 5;
	float subsample_threshold = 0.0001;
	float min_alpha = init_alpha * 0.0001;
	bool cbow_mean = true;
	int negative = 0;
	int num_threads = 1;
	int iter = 1;
	int min_count = 5;

	if ((i = ArgPos((char *)"-size", argc, argv)) > 0)
		word_dim = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0)
		input_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0)
		save_vocab_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0)
		read_vocab_file = std::string(argv[i + 1]);
	//if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-model", argc, argv)) > 0)
		model = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0)
		init_alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0)
		output_file = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0)
		window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-subsample", argc, argv)) > 0)
		subsample_threshold = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-train_method", argc, argv)) > 0)
		train_method = std::string(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0)
		negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0)
		num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-iter", argc, argv)) > 0)
		iter = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0)
		min_count = atoi(argv[i + 1]);

	if(model == "")
	{
		model = "sg";
		cout << "Default use skip gram model" << endl;
	}
	if(train_method == "")
	{
		train_method = "ns";
		cout << "Default use negative sampling model" << endl;
	}

	if(train_method == "ns" && negative <= 0)
	{
		cout << "Please set -negative > 0!" << endl;
		return 1;
	}
	if(train_method == "hs" && negative > 0)
	{
		cout << "Do not set -negative under hierarchical softmax!" << endl;
		return 1;
	}
	if(train_method == "hs" && model.find("align") != string::npos)
	{
		cout << "Please use negative sampling in aligned skip gram model!" << endl;
		return 1;
	}

	if(cbow_mean)
		init_alpha = 0.05;

	Word2Vec w2v(iter, window, min_count, table_size, word_dim, negative, subsample_threshold,
		init_alpha, min_alpha, cbow_mean, num_threads, train_method, model);

	omp_set_num_threads(num_threads);
	//vector<vector<string>> sentences = w2v.line_docs("imdb_train.txt");
	vector<vector<string>> sentences = text8_corpus();
	w2v.build_vocab(sentences);
	w2v.init_weights(w2v.vocab.size());
	if(save_vocab_file != "")
		w2v.save_vocab(save_vocab_file);

	w2v.train(sentences);

	if(output_file != "")
	{
		if(train_method == "hs" && model == "cbow")
			w2v.save_word2vec(output_file, w2v.C);	
		else
			w2v.save_word2vec(output_file, w2v.W);
	}

	return 0;
}