#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>

#include "util.h"
#include "allPhonemeSet.h"
#include "weightWF.h"
#include "param.h"

//extern "C" {
//# include "svm_common.h"
//# include "svm_learn.h"
//} 
#include "svm_common.h"
#include "svm_learn.h"

using namespace std;

class phraseModel
{
	weightWF myWF;
	allPhonemeSet myAllPhoneme;

	// hash map between myFeature and SVMfeature
	hash_string_long featureHash;
	
public:
	phraseModel(void);
	~phraseModel(void);

	void training(param& myParam);
	void testing(param& myParam);

	//void readingAlignedFile(param &myParam, string filename, vector<data>& output);
	void readingAlignedFile(param &myParam, string filename, hash_string_vData& output, bool genFea=false);
	//void dataToUnique(param &myParam, vector<data>& inData, hash_string_vData& outUniqueData);

	void readingTestingFile(param &myParam, string filename, vector_vData& output, bool genFea=false);

	void readingExtraFeature(param &myParam, string filename, hash_string_vData& output);

	//void initialize(param& myParam, vector<data> trainingData);
	void initialize(param& myParam, hash_string_vData& trainDataUnique);

	vector_2str phrasalDecoder(param& myParam, vector_str unAlignedX, 
		vector_2str &alignedXnBest, vector_3str &featureNbest, vector<double>& scoreNbest);

	vector_2str phrasalDecoder_beam(param& myParam, vector_str unAlignedX, 
		vector_2str &alignedXnBest, vector_3str &featureNbest, vector<double>& scoreNbest);

	vector_str ngramFeatureGen(param& myParam, vector_str lcontext, vector_str focus, vector_str rcontext);
	vector_2str genFeature(param& myParam, data dpoint);

	double getLocalFeatureScore(param& myParam, vector_str observation, string classLabel);
	double getOrderFeatureScore(param& myParam, vector_str observation, string p_class, string c_class);
	double getJointGramFeatureScore(param& myParam, vector_str jointX, vector_str jointY, string currentX, string currentY);
	double getJointForwardGramFeatureScore(param& myParam, string currentX, string currentY, vector_str xForward);

	double minEditDistance(vector<string> str1, vector<string> str2, string ignoreString = "");

	long my_feature_hash(string feaList, string phonemeTarget, hash_string_long *featureHash);

	WORD *my_feature_map_word(param &myParam, vector_2str featureList, vector_str alignedTarget, 
		hash_string_long *featureHash, long max_words_doc, vector_str alignedSource);

	void my_feature_hash_avg(param &myParam, vector_2str featureList, vector_str alignedTarget, 
											  hash_string_long *featureHash, double scaleAvg, map<long,double> &idAvg);

	WORD *my_feature_hash_map_word(param &myParam, map<long,double> &idAvg, long max_words_doc);


	double cal_score_candidate(param &myParam, vector_2str featureList, vector_str alignedTarget, vector_str alignedSource);

	double cal_score_hash_avg(param &myParam, map<long,double> &idAvg);

	string my_feature_hash_retrieve(hash_string_long *featureHash, long value);

	void writeMaxPhraseSize(param &myParam, string filename);
	void readMaxPhraseSize(param &myParam, string filename);
};


