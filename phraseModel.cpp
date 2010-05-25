#include "phraseModel.h"

//------- SVM default parameters -------------------//

void set_default_parameters(LEARN_PARM *learn_parm, KERNEL_PARM *kernel_parm)
{
	//strcpy (modelfile, "svm_model");
	strcpy (learn_parm->predfile, "trans_predictions");
	strcpy (learn_parm->alphafile, "");
	//strcpy (restartfile, "");

	learn_parm->biased_hyperplane=1;
	learn_parm->sharedslack=0;
	learn_parm->remove_inconsistent=0;
	learn_parm->skip_final_opt_check=0;
	learn_parm->svm_maxqpsize=10;
	learn_parm->svm_newvarsinqp=0;
	learn_parm->svm_iter_to_shrink=-9999;
	learn_parm->maxiter=100000;
	learn_parm->kernel_cache_size=40;
	learn_parm->svm_c=9999999;
	learn_parm->eps=0.1;
	learn_parm->transduction_posratio=-1.0;
	learn_parm->svm_costratio=1.0;
	learn_parm->svm_costratio_unlab=1.0;
	learn_parm->svm_unlabbound=1E-5;
	learn_parm->epsilon_crit=0.001;
	learn_parm->epsilon_a=1E-15;
	learn_parm->compute_loo=0;
	learn_parm->rho=1.0;
	learn_parm->xa_depth=0;
	kernel_parm->kernel_type=0;
	kernel_parm->poly_degree=3;
	kernel_parm->rbf_gamma=1.0;
	kernel_parm->coef_lin=1;
	kernel_parm->coef_const=1;
	strcpy(kernel_parm->custom,"empty");

	if(learn_parm->svm_iter_to_shrink == -9999) {
    if(kernel_parm->kernel_type == LINEAR) 
      learn_parm->svm_iter_to_shrink=2;
    else
      learn_parm->svm_iter_to_shrink=100;
  }
	if((learn_parm->skip_final_opt_check) 
     && (kernel_parm->kernel_type == LINEAR)) {
    printf("\nIt does not make sense to skip the final optimality check for linear kernels.\n\n");
    learn_parm->skip_final_opt_check=0;
  }    
  if((learn_parm->skip_final_opt_check) 
     && (learn_parm->remove_inconsistent)) {
    printf("\nIt is necessary to do the final optimality check when removing inconsistent \nexamples.\n");
  //  wait_any_key();
  //  print_help();
    exit(0);
  }    
  if((learn_parm->svm_maxqpsize<2)) {
    printf("\nMaximum size of QP-subproblems not in valid range: %ld [2..]\n",learn_parm->svm_maxqpsize); 
  //  wait_any_key();
  //  print_help();
    exit(0);
  }
  if((learn_parm->svm_maxqpsize<learn_parm->svm_newvarsinqp)) {
    printf("\nMaximum size of QP-subproblems [%ld] must be larger than the number of\n",learn_parm->svm_maxqpsize); 
    printf("new variables [%ld] entering the working set in each iteration.\n",learn_parm->svm_newvarsinqp); 
  //  wait_any_key();
  //  print_help();
    exit(0);
  }
  if(learn_parm->svm_iter_to_shrink<1) {
    printf("\nMaximum number of iterations for shrinking not in valid range: %ld [1,..]\n",learn_parm->svm_iter_to_shrink);
  //  wait_any_key();
  //  print_help();
    exit(0);
  }
  if(learn_parm->svm_c<0) {
    printf("\nThe C parameter must be greater than zero!\n\n");
 //   wait_any_key();
 //   print_help();
    exit(0);
  }
  if(learn_parm->transduction_posratio>1) {
    printf("\nThe fraction of unlabeled examples to classify as positives must\n");
    printf("be less than 1.0 !!!\n\n");
 //   wait_any_key();
 //   print_help();
    exit(0);
  }
  if(learn_parm->svm_costratio<=0) {
    printf("\nThe COSTRATIO parameter must be greater than zero!\n\n");
 //   wait_any_key();
 //   print_help();
    exit(0);
  }
  if(learn_parm->epsilon_crit<=0) {
    printf("\nThe epsilon parameter must be greater than zero!\n\n");
 //   wait_any_key();
 //   print_help();
    exit(0);
  }
  if(learn_parm->rho<0) {
    printf("\nThe parameter rho for xi/alpha-estimates and leave-one-out pruning must\n");
    printf("be greater than zero (typically 1.0 or 2.0, see T. Joachims, Estimating the\n");
    printf("Generalization Performance of an SVM Efficiently, ICML, 2000.)!\n\n");
 //   wait_any_key();
 //   print_help();
    exit(0);
  }
  if((learn_parm->xa_depth<0) || (learn_parm->xa_depth>100)) {
    printf("\nThe parameter depth for ext. xi/alpha-estimates must be in [0..100] (zero\n");
    printf("for switching to the conventional xa/estimates described in T. Joachims,\n");
    printf("Estimating the Generalization Performance of an SVM Efficiently, ICML, 2000.)\n");
 //   wait_any_key();
 //   print_help();
    exit(0);
  }
}

//------- SVM default parameters -------------------//

phraseModel::phraseModel(void)
{
}

phraseModel::~phraseModel(void)
{
}

void phraseModel::readingExtraFeature(param &myParam, string filename, hash_string_vData& output)
{
	cout << "Reading file: " << filename << endl;

	ifstream INPUTFILE;

	INPUTFILE.open(filename.c_str());
	if (! INPUTFILE)
	{
		cerr << endl << "Error: unable to open file " << filename << endl;
		exit(-1);
	}

	while (! INPUTFILE.eof())
	{
	}
	
	INPUTFILE.close();
}

void phraseModel::readingTestingFile(param &myParam, string filename, vector_vData& output, bool genFea)
{
	size_t totRead = 0;
	cout << "Reading file: " << filename << endl;
	
	ifstream INPUTFILE;

	INPUTFILE.open(filename.c_str());
	if (! INPUTFILE)
	{
		cerr << endl << "Error: unable to open file " << filename << endl;
		exit(-1);
	}

	while (! INPUTFILE.eof())
	{
		string line;
		vector<string> lineList;

		data lineData;

		getline(INPUTFILE, line);

		// ignore empty line
		if (line == "")
		{
			continue;
		}

		// ignore line that indicate no alignment //
		if (line.find("NO ALIGNMENT") != string::npos)
		{
			continue;
		}

		lineList = splitBySpace(line);

		if (lineList.size() > 4)
		{
			cerr << endl << "Warning: wrong expected format" << endl << line << endl;
		}
		else
		{
			// read aligned data // 
			Tokenize(lineList[0], lineData.alignedX, "|");

			if (lineList.size() > 1)
			{
				Tokenize(lineList[1], lineData.alignedY, "|");
			}

			if (lineList.size() > 2)
			{
				//lineData.alignRank = convertTo<int>(lineList[2]);
				lineData.alignRank = atoi(lineList[2].c_str());
			}
			else
			{
				// default // 
				lineData.alignRank = 1;
			}

			if (lineList.size() > 3)
			{
				//lineData.alignScore = convertTo<double>(lineList[3]);
				lineData.alignScore = atof(lineList[3].c_str());
			}
			else
			{
				lineData.alignScore = 1;
			}

			// re-format to unaligned data //
			for (vector<string>::iterator pos = lineData.alignedX.begin() ; pos != lineData.alignedX.end() ; pos++)
			{
				Tokenize(*pos, lineData.unAlignedX, myParam.inChar);
				int nDelete = removeSubString(*pos, myParam.inChar); // remove inChar
				int phraseSize;
				if (myParam.inChar == "")
				{
					phraseSize = (*pos).size();
				}
				else
				{
					phraseSize = nDelete + 1;
				}
				lineData.phraseSizeX.push_back(phraseSize);

				// re-adjust maxX according to |phrase| //
				if (phraseSize > myParam.maxX)
				{
					myParam.maxX = phraseSize;
				}
			}
			removeVectorElem(lineData.unAlignedX, "_"); // remove null

			if (lineList.size() > 1)
			{
				// re-format to unaligned data //
				for (vector<string>::iterator pos = lineData.alignedY.begin() ; pos != lineData.alignedY.end() ; pos++)
				{
					Tokenize(*pos, lineData.unAlignedY, myParam.inChar);
					removeSubString(*pos, myParam.inChar); // remove inChar
				}
				removeVectorElem(lineData.unAlignedY, "_"); // remove null
			}
			//output.push_back(lineData);

			if (genFea)
			{
				lineData.feaVec = genFeature(myParam, lineData);
			}

			string unAlignedWord = join(lineData.unAlignedX, "", "");

			//output[unAlignedWord].push_back(lineData);
			dataplus dataTMP; 
			dataTMP.mydata.push_back(lineData);
			output.push_back(dataTMP);
			//output[unAlignedWord].mydata.push_back(lineData);
			totRead++;
		}
	}
	INPUTFILE.close();
	cout << "Total read: " << totRead << " instances" << endl;
}

void phraseModel::readingAlignedFile(param &myParam, string filename, hash_string_vData& output, bool genFea)
{
	size_t totRead = 0;
	cout << "Reading file: " << filename << endl;
	
	ifstream INPUTFILE;

	INPUTFILE.open(filename.c_str());
	if (! INPUTFILE)
	{
		cerr << endl << "Error: unable to open file " << filename << endl;
		exit(-1);
	}

	while (! INPUTFILE.eof())
	{
		string line;
		vector<string> lineList;

		data lineData;

		getline(INPUTFILE, line);

		// ignore empty line
		if (line == "")
		{
			continue;
		}

		// ignore line that indicate no alignment //
		if (line.find("NO ALIGNMENT") != string::npos)
		{
			continue;
		}

		lineList = splitBySpace(line);

		// ignore empty line filled with space //
		if (lineList.size() < 1)
		{
			continue;
		}

		if (lineList.size() > 4)
		{
			cerr << endl << "Warning: wrong expected format" << endl << line << endl;
		}
		else
		{
			// read aligned data // 
			Tokenize(lineList[0], lineData.alignedX, "|");

			if (lineList.size() > 1)
			{
				Tokenize(lineList[1], lineData.alignedY, "|");
			}

			if (lineList.size() > 2)
			{
				//lineData.alignRank = convertTo<int>(lineList[2]);
				lineData.alignRank = atoi(lineList[2].c_str());
			}
			else
			{
				// default // 
				lineData.alignRank = 1;
			}

			if (lineList.size() > 3)
			{
				//lineData.alignScore = convertTo<double>(lineList[3]);
				lineData.alignScore = atof(lineList[3].c_str());
			}
			else
			{
				lineData.alignScore = 1;
			}

			// re-format to unaligned data //
			for (vector<string>::iterator pos = lineData.alignedX.begin() ; pos != lineData.alignedX.end() ; pos++)
			{
				Tokenize(*pos, lineData.unAlignedX, myParam.inChar);
				int nDelete = removeSubString(*pos, myParam.inChar); // remove inChar
				int phraseSize;
				if (myParam.inChar == "")
				{
					phraseSize = (*pos).size();
				}
				else
				{
					phraseSize = nDelete + 1;
				}
				lineData.phraseSizeX.push_back(phraseSize);

				// re-adjust maxX according to |phrase| //
				if (phraseSize > myParam.maxX)
				{
					myParam.maxX = phraseSize;
				}
			}
			removeVectorElem(lineData.unAlignedX, "_"); // remove null

			if (lineList.size() > 1)
			{
				// re-format to unaligned data //
				for (vector<string>::iterator pos = lineData.alignedY.begin() ; pos != lineData.alignedY.end() ; pos++)
				{
					Tokenize(*pos, lineData.unAlignedY, myParam.inChar);
					removeSubString(*pos, myParam.inChar); // remove inChar
				}
				removeVectorElem(lineData.unAlignedY, "_"); // remove null
			}
			//output.push_back(lineData);

			if (genFea)
			{
				lineData.feaVec = genFeature(myParam, lineData);
			}

			string unAlignedWord = join(lineData.unAlignedX, "", "");

			//output[unAlignedWord].push_back(lineData);
			output[unAlignedWord].mydata.push_back(lineData);
			totRead++;
		}
	}
	INPUTFILE.close();
	cout << "Total read: " << totRead << " instances" << endl;
}

void phraseModel::initialize(param& myParam, hash_string_vData& trainDataUnique)
{
	for (hash_string_vData::iterator train_pos = trainDataUnique.begin(); train_pos != trainDataUnique.end(); train_pos++)
	{
		//for (unsigned long i = 0; i < train_pos->second.size(); i++)
		for (unsigned long i = 0; i < train_pos->second.mydata.size(); i++)
		{
			//for (unsigned long j = 0; j < train_pos->second[i].alignedX.size() ; j++)
			for (unsigned long j = 0; j < train_pos->second.mydata[i].alignedX.size() ; j++)
			{
				//myAllPhoneme.addPhoneme(train_pos->second[i].alignedY[j], train_pos->second[i].alignedX[j], true);
				myAllPhoneme.addPhoneme(train_pos->second.mydata[i].alignedY[j], train_pos->second.mydata[i].alignedX[j], true);
			}
		}
	}
}

vector_str phraseModel::ngramFeatureGen(param &myParam, vector_str lcontext, vector_str focus, vector_str rcontext)
{
	vector_str output;
	vector_str allSeen;
	string mergeFocus = join(focus,"","");

	if (lcontext.size() < myParam.contextSize)
	{
		lcontext.insert(lcontext.begin(), "{");
	}

	if (rcontext.size() < myParam.contextSize)
	{
		rcontext.push_back("}");
	}

	int posFocus = lcontext.size();

	allSeen.insert(allSeen.end(), lcontext.begin(), lcontext.end()); // left context
	allSeen.push_back(mergeFocus); // focus token
	allSeen.insert(allSeen.end(), rcontext.begin(), rcontext.end()); // right context

	string feaStr;

	for (int i = 0; i < allSeen.size(); i++)
	{
		for (int k = 1; (k <= myParam.nGram) && (k + i <= allSeen.size()); k++)
		{

			feaStr = "L:" + stringify(i - posFocus) + ":" + stringify(i + k - posFocus - 1) + ":";
			feaStr += join(allSeen, i, i + k, "","");

			output.push_back(feaStr);
		} 
	}
	return output;
}

double phraseModel::getLocalFeatureScore(param &myParam, vector_str observation, string classLabel)
{
	double output = 0;
	
	for (vector_str::iterator feaListPos = observation.begin(); feaListPos != observation.end(); feaListPos++)
	{
		output += myWF.getFeature(*feaListPos, classLabel, myParam.atTesting);
	}

	return output;
}

double phraseModel::getOrderFeatureScore(param &myParam, vector_str observation, string p_class, string c_class)
{
	double output = 0;

	if (myParam.markovOrder == 1)
	{
		output += myWF.getFeature("P:-1:" + p_class, c_class, myParam.atTesting);
	}

	if (myParam.linearChain)
	{
		for (vector_str::iterator feaListPos = observation.begin(); feaListPos != observation.end(); feaListPos++)
		{
			output += myWF.getFeature(*feaListPos + "P:-1:" + p_class, c_class, myParam.atTesting);
		}
	}

	return output;
}

double phraseModel::getJointForwardGramFeatureScore(param &myParam, string currentX, string currentY, vector_str xForward)
{
	double output = 0;

	// start to count 1-gram, 2-gram ..., M-gram features //
	// note: if FM=0, it means we don't include the feature //
	// note: if FM=1, it is the unigram. We don't count it here //
	
	if (myParam.jointFMgram <= 1)
		return 0;

	int max_forward = min<int>(myParam.jointFMgram, xForward.size() + 1);

	vector_str yCurrentCandidate;
	vector_str yPastCandidate;

	// too much thinking now: so, for now, it works only single token (no phrasal input) //
	for (int i = 1; i < max_forward; i++)
	{
		yCurrentCandidate = myAllPhoneme.getPhoneme(xForward[i-1], true);
		string feaStrX = join(xForward, 0, i, "-" , "");
		string feaStrY;

		vector_str yKeepHistory;
		for (vector_str::iterator pos = yCurrentCandidate.begin() ; pos != yCurrentCandidate.end(); pos++)
		{
			if ( i-1 > 0)
			{
				for (vector_str::iterator p_pos = yPastCandidate.begin() ; p_pos != yPastCandidate.end(); p_pos++)
				{
					feaStrY = *p_pos + "-" + *pos;
					yKeepHistory.push_back(feaStrY);

					output += myWF.getFeature("JL:1:" + stringify(i) + ":" + feaStrX + "JP:" + feaStrY + "L:" + currentX, 
								currentY, myParam.atTesting);
				}
			}
			else
			{
				feaStrY = *pos;
				yKeepHistory.push_back(feaStrY);

				output += myWF.getFeature("JL:1:" + stringify(i) + ":" + feaStrX + "JP:" + feaStrY + "L:" + currentX, 
							currentY, myParam.atTesting);
			}
		}
		yPastCandidate = yKeepHistory;
	}

	return output;
}

double phraseModel::getJointGramFeatureScore(param &myParam, vector_str jointX, vector_str jointY, string currentX, string currentY)
{
	double output = 0;

	// start to count 1-gram, 2-gram ..., M-gram features //
	// note: if M=0, it means we don't include the jointMgram feature //

	if (myParam.jointMgram <= 0)
		return 0;

	int max_history = min<int>(myParam.jointMgram, (jointX.size() + 1));

	// get score of a unigram (currentX,currentY) //
	output += myWF.getFeature("JL:0:0:JP:L:" + currentX, currentY, myParam.atTesting);

	// get scores of looking back i history;  //
	for (int i = 1; i < max_history; i++)
	{
		for (int j = i ; j > 0; j--)
		{
			string feaStrX = join(jointX, jointX.size() - i, jointX.size() - j + 1, "-", "");
			string feaStrY = join(jointY, jointY.size() - i, jointY.size() - j + 1, "-", "");
		
			output += myWF.getFeature("JL:" + stringify(-i) + ":" + stringify(-j) + ":" + feaStrX + "JP:" + feaStrY + "L:" + currentX, 
			currentY, myParam.atTesting);
		}
	}

	return output;
}

vector_2str phraseModel::phrasalDecoder_beam(param &myParam, vector_str unAlignedX, 
											 vector_2str &alignedXnBest, vector_3str &featureNbest,
											 vector<double> &scoreNbest)
{
	vector_2str nBestOutput;
	vector_3str allFeatureStr;
	DD_btable beamTable;

	beamTable.resize(unAlignedX.size());
	allFeatureStr.resize(unAlignedX.size());

	// go over state //
	for (int i = 0; i < unAlignedX.size(); i++)
	{
		allFeatureStr[i].resize(myParam.maxX + 1);
		// go over phrases //
		for (int k = 1; k <= myParam.maxX ; k++)
		{
			// make sure "stage + phraseSize k" is reachable //
			if ((i+k) > unAlignedX.size())
			{
				continue;
			}

			int lPosMin = i - myParam.contextSize;
			if (lPosMin < 0)
			{
				lPosMin = 0;
			}

			int rPosMax = i + k + myParam.contextSize;
			if (rPosMax > unAlignedX.size())
			{
				rPosMax = unAlignedX.size();
			}
			vector_str focus(unAlignedX.begin() + i , unAlignedX.begin() + i + k);
			vector_str lContext(unAlignedX.begin() + lPosMin, unAlignedX.begin() + i);
			vector_str rContext(unAlignedX.begin() + i + k, unAlignedX.begin() + rPosMax);
			
			vector<string> allCandidate = myAllPhoneme.getPhoneme(join(focus,"",""), true);

			if (allCandidate.size() > myParam.maxCandidateEach)
			{
				myParam.maxCandidateEach = allCandidate.size();
			}

			// default case : add null when |phrase| = 1 and no candidate //
			// to ensure at least one source generates one target (null included)//
			if ((allCandidate.size() == 0 ) && (k == 1))
			{
				allCandidate.push_back("_");
			}
			
			// skip any |phrase| > 1 and no candidate found //
			if (allCandidate.size() == 0)
			{
				continue;
			}

			// all local features at the current phrase and stage //
			vector<string> featureStr;
			if (! myParam.noContextFea)
			{
				// extract n-gram feature // 
				featureStr = ngramFeatureGen(myParam, lContext, focus, rContext);
				allFeatureStr[i][k] = featureStr; // keep k start from 1 (|phrase| >= 1)
			}

			for (vector<string>::iterator c_pos = allCandidate.begin(); c_pos != allCandidate.end(); c_pos++)
			{
				// calculate local features //
				// getLocalFeatureScore contains only context features which
				// basically the function iterates over the vector<string> featureStr 
				// associated with c_pos 

				double localScore = getLocalFeatureScore(myParam, featureStr, *c_pos);
				double transScore;
				double jointMScore;
				double jointFMScore;

				if (myParam.jointFMgram > 1)
				{
					int startPos = i + k;
					int stopPos = i + k + myParam.jointFMgram - 1;
					
					if (startPos > unAlignedX.size()) startPos = unAlignedX.size();
					if (stopPos > unAlignedX.size()) stopPos = unAlignedX.size();

					vector_str xForward(unAlignedX.begin() + startPos, unAlignedX.begin() + stopPos);
					jointFMScore = getJointForwardGramFeatureScore(myParam, join(unAlignedX, i, i + k, "", ""), *c_pos, xForward);
				}
				else
				{
					jointFMScore = 0;
				}
			
				// no previous decision yet //
				if (i == 0)
				{
					// getOrderFeatureScore contains only markov order 1 and linear-chain 
					// In linear-chain, we iterate over featureStr associated with ""(c-1_pos) and c_pos
					// In markove=1, we get score of ""(c-1_pos) and c_pos 
					transScore = getOrderFeatureScore(myParam, featureStr, "", *c_pos);

					btable Btmp;

					Btmp.currentX = join(unAlignedX, i, i + k, "", "");
					Btmp.currentY = *c_pos;
					Btmp.phraseSize.push_back(k);
					Btmp.jointX.clear();
					Btmp.jointY.clear();

					jointMScore = getJointGramFeatureScore(myParam, Btmp.jointX, Btmp.jointY, Btmp.currentX, Btmp.currentY);

					Btmp.score = localScore + transScore + jointMScore + jointFMScore;
					
					// no previous joint-gram history //

					beamTable[i+k-1].push_back(Btmp);
				}
				else // consider previous decision
				{
					for (D_btable::iterator p_pos = beamTable[i-1].begin(); p_pos != beamTable[i-1].end() ; p_pos++)
					{
						transScore = getOrderFeatureScore(myParam, featureStr, p_pos->currentY, *c_pos);

						btable Btmp;
						
						Btmp.phraseSize = p_pos->phraseSize;
						Btmp.phraseSize.push_back(k);

						Btmp.currentX = Btmp.currentX = join(unAlignedX, i, i + k, "", "");
						Btmp.currentY = *c_pos;
						
						Btmp.jointX = p_pos->jointX;
						Btmp.jointY = p_pos->jointY;
						Btmp.jointX.push_back(p_pos->currentX);
						Btmp.jointY.push_back(p_pos->currentY);
						
						jointMScore = getJointGramFeatureScore(myParam, Btmp.jointX, Btmp.jointY, Btmp.currentX, Btmp.currentY);

						Btmp.score = p_pos->score + transScore + localScore + jointMScore + jointFMScore; 

						beamTable[i+k-1].push_back(Btmp);	
					}
				}
			}
		}

		// reducing size to max(n-best,beam_size) //
		D_btable Dbtmp;
		int max_beamSize = max(myParam.nBest, myParam.beamSize);

		Dbtmp = beamTable[i];
		if (Dbtmp.size() > max_beamSize)
		{
			D_btable Dbtmp_sort(max_beamSize);
			partial_sort_copy(Dbtmp.begin(), Dbtmp.end(), Dbtmp_sort.begin(), Dbtmp_sort.end(), DbSortedFn);
			beamTable[i] = Dbtmp_sort;
		}
	}


	// sort score //
	sort(beamTable[unAlignedX.size() - 1].begin(), beamTable[unAlignedX.size()-1].end(), DbSortedFn);	
	
	// backtracking - nbest
	for (int k =0 ; (k < myParam.nBest) && (beamTable[unAlignedX.size() - 1].size() > 0) ; k++)
	{
		vector_str tempBestOutput;
		vector_str tempBestAlignedX;
		vector_2str tempBestFeatureStr;

		// current max element 
		scoreNbest.push_back(beamTable[unAlignedX.size() - 1][0].score);
		
		// current best output
		tempBestOutput = beamTable[unAlignedX.size() - 1][0].jointY;
		tempBestOutput.push_back(beamTable[unAlignedX.size() - 1][0].currentY);

		// current best structure
		tempBestAlignedX = beamTable[unAlignedX.size() - 1][0].jointX;
		tempBestAlignedX.push_back(beamTable[unAlignedX.size() - 1][0].currentX);

		// current best local features
		int sum_phraseSize=0;
		for (int i = 0; i < beamTable[unAlignedX.size() - 1][0].phraseSize.size(); i++)
		{
			int c_phraseSize = beamTable[unAlignedX.size() - 1][0].phraseSize[i];
			tempBestFeatureStr.push_back(allFeatureStr[sum_phraseSize][c_phraseSize]);

			sum_phraseSize += c_phraseSize;
		}
		
		nBestOutput.push_back(tempBestOutput);
		alignedXnBest.push_back(tempBestAlignedX);
		featureNbest.push_back(tempBestFeatureStr);

		// remove the top from the chart //

		beamTable[unAlignedX.size() -1].erase(beamTable[unAlignedX.size() - 1].begin());
	}
	
	return nBestOutput;
}

vector_2str phraseModel::phrasalDecoder(param &myParam, vector_str unAlignedX, 
										vector_2str &alignedXnBest, vector_3str &featureNbest,
										vector<double> &scoreNbest)
{
	vector_2str nBestOutput;
	vector_3str allFeatureStr;
	D_hash_string_qtable Q;

	Q.resize(unAlignedX.size());
	allFeatureStr.resize(unAlignedX.size());

	// go over state //
	for (int i = 0; i < unAlignedX.size(); i++)
	{
		allFeatureStr[i].resize(myParam.maxX + 1);
		// go over phrases //
		for (int k = 1; k <= myParam.maxX ; k++)
		{
			if ((i + k) > unAlignedX.size())
			{
				continue;
			}

			int lPosMin = i - myParam.contextSize;
			if (lPosMin < 0)
			{
				lPosMin = 0;
			}

			int rPosMax = i + k + myParam.contextSize;
			if (rPosMax > unAlignedX.size())
			{
				rPosMax = unAlignedX.size();
			}
			vector_str focus(unAlignedX.begin() + i , unAlignedX.begin() + i + k);
			vector_str lContext(unAlignedX.begin() + lPosMin, unAlignedX.begin() + i);
			vector_str rContext(unAlignedX.begin() + i + k, unAlignedX.begin() + rPosMax);
			
			vector<string> allCandidate = myAllPhoneme.getPhoneme(join(focus,"",""), true);

			// default case : add null when |phrase| = 1 and no candidate //
			// to ensure at least one source generates one target (null included)//
			if ((allCandidate.size() == 0 ) && (k == 1))
			{
				allCandidate.push_back("_");
			}
			
			// skip any |phrase| > 1 and no candidate found //
			if (allCandidate.size() == 0)
			{
				continue;
			}

			// extract n-gram feature // 
			vector<string> featureStr;
			featureStr = ngramFeatureGen(myParam, lContext, focus, rContext);

			allFeatureStr[i][k] = featureStr; // keep k start from 1 (|phrase| >= 1)

			for (vector<string>::iterator c_pos = allCandidate.begin(); c_pos != allCandidate.end(); c_pos++)
			{
				double localScore = getLocalFeatureScore(myParam, featureStr, *c_pos);
				double transScore;
			
				// no previous decision yet //
				if (i == 0)
				{
					transScore = getOrderFeatureScore(myParam, featureStr, "", *c_pos);

					qtable Qtmp;

					Qtmp.score = localScore + transScore;
					Qtmp.phraseSize = k;
					Qtmp.backTracking = "";
					Qtmp.backRanking = -1;
					Qtmp.backQ = -1;

					Q[i+k-1][*c_pos].push_back(Qtmp);
				}
				else // consider previous decision
				{
					for (hash_string_Dqtable::iterator p_pos = Q[i-1].begin(); p_pos != Q[i-1].end(); p_pos++)
					{
						transScore = getOrderFeatureScore(myParam, featureStr, p_pos->first, *c_pos);

						//// debug //
						//if (p_pos->second.size() > 10)
						//{
						//	int ssi = p_pos->second.size();
						//	string ssp = p_pos->first;
						//	D_qtable ssQ = p_pos->second;
						//}

						for (unsigned int r = 0 ; r < p_pos->second.size(); r++)
						{
							qtable Qtmp;

							Qtmp.score = p_pos->second[r].score + transScore + localScore;
							Qtmp.phraseSize = k;
							Qtmp.backTracking  = p_pos->first;
							Qtmp.backRanking = r;
							Qtmp.backQ = i-1;
							
							Q[i+k-1][*c_pos].push_back(Qtmp);

						}	
					}
				}
			}
		}

		// BIG possible bug here .. Q[i] Vs. Q[i-1] .. why did it still work?
		//for (hash_string_Dqtable::iterator c_pos = Q[i].begin(); c_pos != Q[i-1].end(); c_pos++)
		for (hash_string_Dqtable::iterator c_pos = Q[i].begin(); c_pos != Q[i].end(); c_pos++)
		{
			// reducing size to n-best
			D_qtable Dqtmp;

			Dqtmp = Q[i][c_pos->first];

			if (Dqtmp.size() > myParam.nBest)
			{
				D_qtable Dqtmp_sort(myParam.nBest);
				partial_sort_copy(Dqtmp.begin(), Dqtmp.end(), Dqtmp_sort.begin(), Dqtmp_sort.end(), DqSortedFn);
				Q[i][c_pos->first] = Dqtmp_sort;
			}
		}
	}

	// sort score //
	for (hash_string_Dqtable::iterator pos = Q[unAlignedX.size() - 1].begin(); pos != Q[unAlignedX.size() - 1].end(); pos++)
	{
		sort(pos->second.begin(), pos->second.end(), DqSortedFn);
	}	
	
	// backtracking - nbest
	for (int k =0 ; (k < myParam.nBest) && (Q[unAlignedX.size() - 1].size() > 0) ; k++)
	{
		vector_str tempBestOutput;
		vector_str tempBestAlignedX;
		vector_2str tempBestFeatureStr;

		double max_score = -10e20;
		string max_pos = "";
		
		
		// find the max score //
		for (hash_string_Dqtable::iterator pos = Q[unAlignedX.size() - 1].begin(); pos != Q[unAlignedX.size() - 1].end(); pos++)
		{
			double score_candidate = pos->second[0].score;

			if (score_candidate > max_score)
			{
				max_score = score_candidate;
				max_pos = pos->first;
			}
			else if (score_candidate == max_score)
			{
				if (pos->first > max_pos)
				{
					max_score = score_candidate;
					max_pos = pos->first;
				}
			}
		}

		if (max_score <= -10e20)
		{
			cout << "Can't find any candidate (buggy!!)" << endl;
			exit(-1);
		}

		scoreNbest.push_back(max_score);
		string last_element = max_pos;
		int last_rank = 0;
		int last_phraseSize = Q[unAlignedX.size() -1][last_element][last_rank].phraseSize;

		tempBestOutput.push_back(last_element);
		tempBestAlignedX.push_back(join(unAlignedX,unAlignedX.size() - last_phraseSize, unAlignedX.size(), "", ""));
		tempBestFeatureStr.push_back(allFeatureStr[unAlignedX.size() - last_phraseSize][last_phraseSize]);
		

		int i = unAlignedX.size() - 1;

		while (i > -1)
		{
			qtable Qtmp = Q[i][last_element][last_rank];
			string last_element_tmp = Qtmp.backTracking;
			int last_rank_tmp = Qtmp.backRanking;
			i = Qtmp.backQ;

			if (i > -1)
			{
				last_phraseSize = Q[i][last_element_tmp][last_rank_tmp].phraseSize;
				tempBestFeatureStr.push_back(allFeatureStr[i - last_phraseSize + 1][last_phraseSize]);
				tempBestAlignedX.push_back(join(unAlignedX,i - last_phraseSize + 1, i + 1, "",""));
				tempBestOutput.push_back(last_element_tmp);			
			}

			last_element = last_element_tmp;
			last_rank = last_rank_tmp;
		}

		reverse(tempBestOutput.begin(), tempBestOutput.end());
		reverse(tempBestAlignedX.begin(), tempBestAlignedX.end());
		reverse(tempBestFeatureStr.begin(), tempBestFeatureStr.end());

		nBestOutput.push_back(tempBestOutput);
		alignedXnBest.push_back(tempBestAlignedX);
		featureNbest.push_back(tempBestFeatureStr);

		// remove the top from the chart //

		D_qtable D_Qtmp;
		D_Qtmp = Q[unAlignedX.size()-1][max_pos];
		D_Qtmp.erase(D_Qtmp.begin());
		if (D_Qtmp.size() == 0)
		{
			Q[unAlignedX.size()-1].erase(max_pos);
		}
		else
		{
			Q[unAlignedX.size()-1][max_pos] = D_Qtmp;
		}
	}

	return nBestOutput;
}

double phraseModel::minEditDistance(vector<string> str1, vector<string> str2, string ignoreString)
{
	double distanceRate;
	double maxLength;
	double cost;

//	string str1,str2;

	vector<vector<double> > distance;

	if (ignoreString != "")
	{
		removeVectorElem(str1, ignoreString);
		removeVectorElem(str2, ignoreString);
	}

	//str1 = join(vstr1, "", ignoreString);
	//str2 = join(vstr2, "", ignoreString);

	//Initialize distance matrix
	distance.assign(str1.size()+1, vector<double>(str2.size()+1, 0));

	for (unsigned int i = 0; i < str1.size()+1; i++)
	{
		distance[i][0] = i;
	}
	for (unsigned int j = 0; j < str2.size()+1; j++)
	{
		distance[0][j] = j;
	}

	for (unsigned int i = 0; i < str1.size(); i++)
	{
		for (unsigned int j = 0; j < str2.size(); j++)
		{
			if (str1[i] == str2[j])
			{
				cost = 0;
			}
			else
			{
				cost = 1;
			}

			distance[i+1][j+1] = distance[i][j] + cost;

			if (distance[i+1][j+1] > (distance[i+1][j] + 1))
			{
				distance[i+1][j+1] = distance[i+1][j] + 1;
			}

			if (distance[i+1][j+1] > (distance[i][j+1] + 1))
			{
				distance[i+1][j+1] = distance[i][j+1] + 1;
			}
		}
	}

	distanceRate = distance[str1.size()][str2.size()];
	maxLength = (double)(max(str1.size(), str2.size()));
	distanceRate = distanceRate / maxLength;

	return distanceRate;

}

long phraseModel::my_feature_hash(string feaList, string phonemeTarget, hash_string_long *featureHash)
{
	long value;

	string key = feaList + "T:" + phonemeTarget;
	hash_string_long::iterator m_pos;

	m_pos = featureHash->find(key);
	if ( m_pos != featureHash->end() )
	{
		return m_pos->second;
	}
	else
	{
		value = featureHash->size() + 1;
		featureHash->insert(make_pair(key,value));
		return value;
	}
}


vector_2str phraseModel::genFeature(param &myParam, data dpoint)
{
	vector_2str output;

	int j = 0;
	for (int i = 0; i < dpoint.unAlignedX.size(); )
	{
		int k = dpoint.phraseSizeX[j];
		int lPosMin = i - myParam.contextSize;
		if (lPosMin < 0)
		{
			lPosMin = 0;
		}

		int rPosMax = i + k + myParam.contextSize;
		if (rPosMax > dpoint.unAlignedX.size())
		{
			rPosMax = dpoint.unAlignedX.size();
		}

		vector_str focus(dpoint.unAlignedX.begin() + i , dpoint.unAlignedX.begin() + i + k);
		vector_str lContext(dpoint.unAlignedX.begin() + lPosMin, dpoint.unAlignedX.begin() + i);
		vector_str rContext(dpoint.unAlignedX.begin() + i + k, dpoint.unAlignedX.begin() + rPosMax);

		if (! myParam.noContextFea)
		{
			output.push_back(ngramFeatureGen(myParam, lContext, focus, rContext));
		}

		i += k;
		j++;
	}
	
	return output;
}

void phraseModel::my_feature_hash_avg(param &myParam, vector_2str featureList, vector_str alignedTarget, 
											  hash_string_long *featureHash, double scaleAvg, map<long,double> &idAvg)
{

	// haven't implement this part to incorporate the jointMgram feature yet //
	for (unsigned int i = 0; i < alignedTarget.size(); i++)
	{
		for (vector_str::iterator feaListPos = featureList[i].begin(); feaListPos != featureList[i].end(); feaListPos++)
		{
			//idSet.push_back(my_feature_hash(*feaListPos, alignedTarget[i], featureHash));
			idAvg[my_feature_hash(*feaListPos, alignedTarget[i], featureHash)] += scaleAvg;

			if (myParam.linearChain)
			{
				if (i == 0)
				{
					//idSet.push_back(my_feature_hash(*feaListPos + "P:-1:", alignedTarget[i], featureHash));
					idAvg[my_feature_hash(*feaListPos + "P:-1:", alignedTarget[i], featureHash)] += scaleAvg;
				}
				else
				{
					//idSet.push_back(my_feature_hash(*feaListPos + "P:-1:" + alignedTarget[i-1], alignedTarget[i], featureHash));
					idAvg[my_feature_hash(*feaListPos + "P:-1:" + alignedTarget[i-1], alignedTarget[i], featureHash)] += scaleAvg;
				}
			}
		}

		// 1st order markov features
		if (myParam.markovOrder == 1)
		{
			if (i == 0)
			{
				//idSet.push_back(my_feature_hash("P:-1:",alignedTarget[i],featureHash));
				idAvg[my_feature_hash("P:-1:",alignedTarget[i],featureHash)] += scaleAvg;
			}
			else
			{
				//idSet.push_back(my_feature_hash("P:-1:" + alignedTarget[i-1], alignedTarget[i], featureHash));
				idAvg[my_feature_hash("P:-1:" + alignedTarget[i-1], alignedTarget[i], featureHash)] += scaleAvg;
			}
		}
	}
}

WORD *phraseModel::my_feature_hash_map_word(param &myParam, map<long,double> &idAvg, long max_words_doc)
{
	long wpos = 0;
	WORD *outWORD = (WORD *) my_malloc(sizeof(WORD) * (max_words_doc + 10));

	for(map<long,double>::iterator pos = idAvg.begin(); pos != idAvg.end(); )
	{
		outWORD[wpos].wnum = pos->first;
		outWORD[wpos].weight = pos->second;
		pos++;

		while ((pos != idAvg.end()) && (pos->first == outWORD[wpos].wnum))
		{
			outWORD[wpos].weight += pos->second;
			pos++;
		}
		wpos++;
	}

	outWORD[wpos].wnum = 0;
	outWORD[wpos].weight = 0;

	return outWORD;

}

WORD *phraseModel::my_feature_map_word(param &myParam, vector_2str featureList, vector_str alignedTarget, 
									   hash_string_long *featureHash, long max_words_doc, vector_str alignedSource)
{
	long wpos = 0;
	WORD *outWORD = (WORD *) my_malloc(sizeof(WORD) * (max_words_doc + 10));
	vector<long> idSet;

	vector_str jointX;
	vector_str jointY;

	for (unsigned int i = 0; i < alignedTarget.size(); i++)
	{
		// when we do include context features //
		if (featureList.size() > 0)
		{
			for (vector_str::iterator feaListPos = featureList[i].begin(); feaListPos != featureList[i].end(); feaListPos++)
			{
				idSet.push_back(my_feature_hash(*feaListPos, alignedTarget[i], featureHash));

				if (myParam.linearChain)
				{
					if (i == 0)
					{
						idSet.push_back(my_feature_hash(*feaListPos + "P:-1:", alignedTarget[i], featureHash));
					}
					else
					{
						idSet.push_back(my_feature_hash(*feaListPos + "P:-1:" + alignedTarget[i-1], alignedTarget[i], featureHash));
					}
				}
			}
		}

		// 1st order markov features
		if (myParam.markovOrder == 1)
		{
			if (i == 0)
			{
				idSet.push_back(my_feature_hash("P:-1:",alignedTarget[i],featureHash));
			}
			else
			{
				idSet.push_back(my_feature_hash("P:-1:" + alignedTarget[i-1], alignedTarget[i], featureHash));
			}
		}

		// jointMgram features
		if (myParam.jointMgram > 0)
		{
			int max_history = min<int>(myParam.jointMgram, (jointX.size() + 1));

			// a unigram (currentX,currentY) //
			idSet.push_back(my_feature_hash("JL:0:0:JP:L:" + alignedSource[i], alignedTarget[i], featureHash));

			// get scores of looking back i history;  //
			for (int j = 1; j < max_history; j++)
			{
				for (int k = j; k > 0; k--)
				{
					string feaStrX = join(jointX, jointX.size() - j, jointX.size() - k + 1, "-", "");
					string feaStrY = join(jointY, jointY.size() - j, jointY.size() - k + 1, "-", "");

					idSet.push_back(my_feature_hash("JL:" + stringify(-j) + ":" + stringify(-k) + ":" + feaStrX + "JP:" + feaStrY + "L:" + alignedSource[i], 
						alignedTarget[i], featureHash));				
				}
			}

			jointX.push_back(alignedSource[i]);
			jointY.push_back(alignedTarget[i]);
		}

		// most likely to be correct only 1-2 situation .. , for now. :-(
		if (myParam.jointFMgram > 1)
		{
			int startPos = i + 1;
			int stopPos = i + 1 + myParam.jointFMgram - 1;
					
			if (startPos > alignedSource.size()) startPos = alignedSource.size();
			if (stopPos > alignedSource.size()) stopPos = alignedSource.size();

			vector_str xForward(alignedSource.begin() + startPos, alignedSource.begin() + stopPos);
			int max_forward = min<int>(myParam.jointFMgram, xForward.size() + 1);

			vector_str yCurrentCandidate;
			vector_str yPastCandidate;

			// too much thinking now: so, for now, it works only single token (no phrasal input) //
			for (int ii = 1; ii < max_forward; ii++)
			{
				yCurrentCandidate = myAllPhoneme.getPhoneme(xForward[ii-1], true);
				string feaStrX = join(xForward, 0, ii, "-" , "");
				string feaStrY;

				vector_str yKeepHistory;
				for (vector_str::iterator pos = yCurrentCandidate.begin() ; pos != yCurrentCandidate.end(); pos++)
				{
					if ( ii-1 > 0)
					{
						for (vector_str::iterator p_pos = yPastCandidate.begin() ; p_pos != yPastCandidate.end(); p_pos++)
						{
							feaStrY = *p_pos + "-" + *pos;
							yKeepHistory.push_back(feaStrY);

							idSet.push_back(my_feature_hash("JL:1:" + stringify(ii) + ":" + feaStrX + "JP:" + feaStrY + "L:" + alignedSource[i], 
								alignedTarget[i], featureHash));
						}
					}
					else
					{
						feaStrY = *pos;
						yKeepHistory.push_back(feaStrY);

						idSet.push_back(my_feature_hash("JL:1:" + stringify(ii) + ":" + feaStrX + "JP:" + feaStrY + "L:" + alignedSource[i], 
								alignedTarget[i], featureHash));
					}
				}
				yPastCandidate = yKeepHistory;
			}
		}
	}

	// convert idSet to WORD
	sort(idSet.begin(),idSet.end());
	for (vector<long>::iterator pos = idSet.begin(); pos != idSet.end(); )
	{
		outWORD[wpos].wnum = *pos;
		outWORD[wpos].weight = 1.0;
		pos++;

		while ((pos != idSet.end()) && (*pos == outWORD[wpos].wnum))
		{
			outWORD[wpos].weight += 1;
			pos++;
		}

		wpos++;
	}
	outWORD[wpos].wnum = 0;
	outWORD[wpos].weight = 0;

	return outWORD;
}

double phraseModel::cal_score_hash_avg(param &myParam, map<long,double> &idAvg)
{
	double score = 0;

	for (map<long,double>::iterator pos = idAvg.begin(); pos != idAvg.end(); pos++)
	{
		score += myWF.getFeature(my_feature_hash_retrieve(&featureHash, pos->first), myParam.atTesting) * pos->second;
	}

	return score;
}

double phraseModel::cal_score_candidate(param &myParam, vector_2str featureList, vector_str alignedTarget, vector_str alignedSource)
{
	double score = 0;

	vector_str jointX;
	vector_str jointY;

	for (int i = 0; i < alignedTarget.size(); i++)
	{
		// make sure we have featureList. It's empty when we don't use contextFeature
		if (featureList.size() > 0)
		{
			score += getLocalFeatureScore(myParam, featureList[i], alignedTarget[i]);

			if (i == 0)
			{
				score += getOrderFeatureScore(myParam, featureList[i], "", alignedTarget[i]);
			}
			else
			{
				score += getOrderFeatureScore(myParam, featureList[i], alignedTarget[i-1], alignedTarget[i]);
			}
		}

		// if we include jointMgram feature, M > 0 
		if (myParam.jointMgram > 0)
		{
			
			score += getJointGramFeatureScore(myParam, jointX, jointY, alignedSource[i], alignedTarget[i]);
			
			jointX.push_back(alignedSource[i]);
			jointY.push_back(alignedTarget[i]);
		}

		// most likely to be correct only 1-2 situation .. , for now. :-(
		if (myParam.jointFMgram > 1)
		{
			int startPos = i + 1;
			int stopPos = i + 1 + myParam.jointFMgram - 1;
					
			if (startPos > alignedSource.size()) startPos = alignedSource.size();
			if (stopPos > alignedSource.size()) stopPos = alignedSource.size();

			vector_str xForward(alignedSource.begin() + startPos, alignedSource.begin() + stopPos);

			score += getJointForwardGramFeatureScore(myParam, alignedSource[i], alignedTarget[i], xForward);
		}
	}

	return score;
}

string phraseModel::my_feature_hash_retrieve(hash_string_long *featureHash, long value)
{
	hash_string_long::iterator m_pos;
	
	m_pos = find_if(featureHash->begin(), featureHash->end(), value_equals<string, long>(value));
	
	if (m_pos != featureHash->end())
	{
		return m_pos->first;
	}
	else
	{
		cout << "ERROR: Can't find matching feature given map value: " << value << endl << endl;
		exit(-1);
	}
}

void phraseModel::readMaxPhraseSize(param &myParam, string filename)
{
	ifstream FILE;
	FILE.open(filename.c_str());

	if (! FILE)
	{
		cerr << "ERROR: Can't open file : " << filename << endl;
	}
	else
	{
		string line;
		getline(FILE,line);

		myParam.maxX = convertTo<int>(line);
	}
	FILE.close();
}
void phraseModel::writeMaxPhraseSize(param &myParam, string filename)
{
	ofstream FILE;

	FILE.open(filename.c_str(),ios_base::trunc);

	if (! FILE)
	{
		cerr << "ERROR: Can't write file : " << filename << endl;
		exit(-1);
	}
	else
	{
		FILE << myParam.maxX << endl;
	}
	FILE.close();
}

//void phraseModel::dataToUnique(param &myParam, vector<data> &inData, hash_string_vData &outUniqueData)
//{
//	// go over data //
//
//	for (long int i = 0; i < inData.size(); i++)
//	{
//		string unAlignedWord = join(inData[i].unAlignedX, "", "");
//		hash_string_vData::iterator it = outUniqueData.find(unAlignedWord);
//
//		if (it != outUniqueData.end())
//		{
//			vector<data> it_vData;
//
//			it_vData = outUniqueData[unAlignedWord];
//			it_vData.push_back(inData[i]);
//
//			outUniqueData[unAlignedWord] = it_vData;
//		}
//		else
//		{
//			vector<data> it_vData;
//			it_vData.push_back(inData[i]);
//
//			outUniqueData[unAlignedWord] = it_vData;
//		}
//	}
//}


void phraseModel::training(param& myParam)
{
	hash_string_vData trainUnique, devUnique;
	unsigned long actual_train, actual_dev;

	cout << endl << "Training starts " << endl;

	cout << "Read training file" << endl;
	readingAlignedFile(myParam, myParam.trainingFile, trainUnique, false);
	cout << "Max source phrase size : " << myParam.maxX << endl;
	cout << "Total number of unique training instances : " << trainUnique.size() << endl;

	// Initialize limited source-target set //
	initialize(myParam, trainUnique);

	if (myParam.devFile != "")
	{
		cout << "Read dev file" << endl;
		readingAlignedFile(myParam, myParam.devFile, devUnique);
		cout << "Total number of unique dev instances : " << devUnique.size() << endl;
	}

	int iter = 0;
	bool stillTrain = true;
	double error;
	double allPhonemeTrain;
	double p_error = 10e6; //initialized error history
	double p_error_train = 10e6; //initialized error history for training
	
	while (stillTrain)
	{
		iter++;
		error = 0;
		allPhonemeTrain = 0;
		actual_train = 0;

		// at training time 
		myParam.atTesting = false;

		cout << endl << "Iteration : " << iter << endl;

		// train over all training data // 
		for (hash_string_vData::iterator train_pos = trainUnique.begin(); train_pos != trainUnique.end(); train_pos++)
		{
			vector_2str nBestAnswer;
			vector_2str alignedXnBest;
			vector_3str featureNbest;
			vector<double> scoreNbest;
			vector<double> nBestPER;

			actual_train++;

			//dev
			//cout << actual_train << endl;
			myParam.maxCandidateEach = 0;
			if (myParam.useBeam)
			{
				nBestAnswer = phrasalDecoder_beam(myParam, train_pos->second.mydata[0].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
			}
			else
			{
				nBestAnswer = phrasalDecoder(myParam, train_pos->second.mydata[0].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
			}

			dataplus multipleRefs = train_pos->second;

			vector<double>  allNbestPER(nBestAnswer.size());
			vector<int> refForNBest(nBestAnswer.size());

			//cout << actual_train << endl;
			// calculate PER for nBest answers respecting to multiple answers
			for (int nbi =0; nbi < nBestAnswer.size(); nbi++)
			{
				double PERjudge; 
				
				if (myParam.alignLoss == "minL")
				{
					PERjudge = 1e9;
				}
				else if (myParam.alignLoss == "maxL")
				{
					PERjudge = -1e9;
				}
				else
				{
					PERjudge = 0;
				}

				if (myParam.alignLoss == "mulA")
				{
					// all y are the same so, it shouldn't give differents in loss(y, y') //
					double pos_PER = minEditDistance(multipleRefs.mydata[0].alignedY, nBestAnswer[nbi], "_");

					allNbestPER[nbi] = pos_PER; 
					refForNBest[nbi] = 0; // just default, shouldn't use this // 

				}
				else if ((myParam.alignLoss != "minS") && (myParam.alignLoss != "maxS"))
				{
					int takingPos = 0;
					double sumAlignScore = 0;
					// go over multiple goal references //
					for (int rti = 0; rti < multipleRefs.mydata.size(); rti++)
					{
						double pos_PER = minEditDistance(multipleRefs.mydata[rti].alignedY, nBestAnswer[nbi], "_");
						

						if (myParam.alignLoss == "minL")
						{
							if (pos_PER < PERjudge)
							{
								PERjudge = pos_PER;
								takingPos = rti;
								sumAlignScore = 1.0;
							}
						}
						else if (myParam.alignLoss == "maxL")
						{
							if (pos_PER > PERjudge)
							{
								PERjudge = pos_PER;
								takingPos = rti;
								sumAlignScore = 1.0;
							}
						}
						else if (myParam.alignLoss == "avgL")
						{
							PERjudge += pos_PER;
							sumAlignScore += 1.0;
						}
						else if (myParam.alignLoss == "ascL")
						{
							PERjudge += (1.0 / multipleRefs.mydata[rti].alignScore) * pos_PER;
							sumAlignScore += (1.0 / multipleRefs.mydata[rti].alignScore);
						}
						else if (myParam.alignLoss == "rakL")
						{
							PERjudge += (1.0 / multipleRefs.mydata[rti].alignRank) * pos_PER;
							sumAlignScore += (1.0 / multipleRefs.mydata[rti].alignRank);
						}
						else
						{
							cerr << "Can't file  " + myParam.alignLoss + " alignLoss handler" << endl; 
							exit(-1);
						}
					}
					allNbestPER[nbi] = (PERjudge / sumAlignScore);
					refForNBest[nbi] = (takingPos);
				}
			}

			// calculate max_words_dos for SVM feature creation //
			long max_words_doc = 0;
			long max_num_features =  (myParam.nGram + 1) * (myParam.nGram + 1) / 2;
			max_words_doc += (max_num_features + myParam.markovOrder + (myParam.jointMgram * myParam.jointMgram)) * 
				(train_pos->second.mydata[0].unAlignedX.size() + 2);

			if (myParam.jointFMgram > 1)
			{
				max_words_doc += long (pow(myParam.maxCandidateEach, (myParam.jointFMgram - 1))) * 
					(train_pos->second.mydata[0].unAlignedX.size() + 2);
			}

			if (myParam.linearChain)
			{
				max_words_doc += ( max_num_features + 1 ) * (train_pos->second.mydata[0].unAlignedX.size() + 2);
			}

			// SVM feature vectors, docs, rhs definitions //
			WORD *xi_yi;

			DOC **docs;
			double *rhs;

			// vector xi_yi //
			hash_string_SVECTOR vector_xi_yi;

			// score of yi //
			hash_string_double score_yi;

			// clear a mapping between featureString <-> featureSVM //
			featureHash.clear();

			// finding score of correct yi , vector_xi_yi
			if ((myParam.alignLoss == "minL") || (myParam.alignLoss == "maxL"))
			{
				for (int nbi = 0; nbi < nBestAnswer.size() ; nbi++)
				{
					// if it doesn't have score for yi, calculate; otherwise skip //
					// indexed by refForNBest[i] //
					if (score_yi.find(stringify(refForNBest[nbi])) == score_yi.end())
					{
						vector_2str feaVector = genFeature(myParam, multipleRefs.mydata[refForNBest[nbi]]);
						double score = cal_score_candidate(myParam, feaVector, multipleRefs.mydata[refForNBest[nbi]].alignedY, multipleRefs.mydata[refForNBest[nbi]].alignedX);

						score_yi[stringify(refForNBest[nbi])] = score;

						xi_yi = my_feature_map_word(myParam, feaVector, multipleRefs.mydata[refForNBest[nbi]].alignedY, &featureHash, max_words_doc, multipleRefs.mydata[refForNBest[nbi]].alignedX);
						
						// vector_xi_yi indexed by refForNBest[i] //
						vector_xi_yi[stringify(refForNBest[nbi])] = create_svector(xi_yi,"",1);				
						free(xi_yi);
					}
				}
			} else if ((myParam.alignLoss == "minS") || (myParam.alignLoss == "maxS"))
			{
				hash_string_double::iterator maxMinElement;
				
				for (int rti = 0; rti < multipleRefs.mydata.size(); rti++)
				{
					vector_2str feaVector = genFeature(myParam, multipleRefs.mydata[rti]);
					double score = cal_score_candidate(myParam, feaVector, multipleRefs.mydata[rti].alignedY, multipleRefs.mydata[rti].alignedX);

					score_yi[stringify(rti)] = score;
				}
				
				// finding the max or min score //
				if (myParam.alignLoss == "minS"){
					maxMinElement = min_element(score_yi.begin(), score_yi.end());
				}
				else
				{
					maxMinElement = max_element(score_yi.begin(), score_yi.end());
				}

				vector_2str feaVector = genFeature(myParam, multipleRefs.mydata[convertTo<int>(maxMinElement->first)]);
				xi_yi = my_feature_map_word(myParam, feaVector, multipleRefs.mydata[convertTo<int>(maxMinElement->first)].alignedY, &featureHash, max_words_doc, multipleRefs.mydata[convertTo<int>(maxMinElement->first)].alignedX);
				vector_xi_yi[maxMinElement->first] = create_svector(xi_yi,"",1);				
				free(xi_yi);

				// go over nbest to assign allNbestPER, refForNBest, vector_xi_yi
				for (int nbi = 0; nbi < nBestAnswer.size(); nbi++)
				{
					refForNBest[nbi] = convertTo<int>(maxMinElement->first);
					allNbestPER[nbi] = minEditDistance(multipleRefs.mydata[refForNBest[nbi]].alignedY, nBestAnswer[nbi], "_");
					
				}
			} else if ((myParam.alignLoss == "avgL") || (myParam.alignLoss == "ascL") || (myParam.alignLoss == "rakL") )
			{
				// CHECK THIS PORTION .. possible bugs , cal_score_hash_avg and featureHash //

				// if it doesn't contain any before //
				if (multipleRefs.idAvg.size() == 0)
				{
					// calculate the average vector //
					map<long,double> idAvg;

					// calculate denominator // 
					double sumAlignScore = 0;
					for (int rti = 0; rti < multipleRefs.mydata.size(); rti++)
					{
						if (myParam.alignLoss == "avgL")
						{
							sumAlignScore += 1.0;
						}
						else if (myParam.alignLoss == "ascL")
						{
							sumAlignScore += (1.0 / multipleRefs.mydata[rti].alignScore);
						}
						else if (myParam.alignLoss == "rakL")
						{
							sumAlignScore += (1.0 / multipleRefs.mydata[rti].alignRank);
						}
						else
						{
							cerr << "Can't file  " + myParam.alignLoss + " alignLoss handler" << endl; 
							exit(-1);
						}
					}

					for (int rti = 0; rti < multipleRefs.mydata.size(); rti++)
					{	
						vector_2str feaVector = genFeature(myParam, multipleRefs.mydata[rti]);
						
						double scaleAvg;
						if (myParam.alignLoss == "avgL")
						{
							scaleAvg = 1.0;
						}
						else if (myParam.alignLoss == "ascL")
						{
							scaleAvg = (1.0 / multipleRefs.mydata[rti].alignScore);
						}
						else if (myParam.alignLoss == "rakL")
						{
							scaleAvg = (1.0 / multipleRefs.mydata[rti].alignRank);
						}
						else
						{
							cerr << "Can't file  " + myParam.alignLoss + " alignLoss handler" << endl; 
							exit(-1);
						}

						scaleAvg /= sumAlignScore;
						my_feature_hash_avg(myParam, feaVector, multipleRefs.mydata[rti].alignedY, &featureHash, scaleAvg, idAvg);
					}
					// get the average idAvg to create the vector of xi_yi and score 
					xi_yi = my_feature_hash_map_word(myParam, idAvg, max_words_doc);
					vector_xi_yi["avg"] = create_svector(xi_yi,"",1);		// average vector //	
					free(xi_yi);

					// save average vector back to xi_yi //
					// can't save the idAvg unless we keep featureHash .. 
					// but keeping featureHash all the way is very expensive 
					
					//train_pos->second.idAvg = idAvg;

					score_yi["avg"] = cal_score_hash_avg(myParam, idAvg);
				}
				else
				{
					// when we have the vector xi_yi already //
					// calculate score //

					xi_yi = my_feature_hash_map_word(myParam, multipleRefs.idAvg, max_words_doc);
					vector_xi_yi["avg"] = create_svector(xi_yi,"",1);
					free(xi_yi);

					score_yi["avg"] = cal_score_hash_avg(myParam, multipleRefs.idAvg);
				}
			}
			else if (myParam.alignLoss == "mulA")
			{
				for (int rti = 0; rti < multipleRefs.mydata.size(); rti++)
				{
					vector_2str feaVector = genFeature(myParam, multipleRefs.mydata[rti]);
					double score = cal_score_candidate(myParam, feaVector, multipleRefs.mydata[rti].alignedY, multipleRefs.mydata[rti].alignedX);

					score_yi[stringify(rti)] = score;

					xi_yi = my_feature_map_word(myParam, feaVector, multipleRefs.mydata[rti].alignedY, &featureHash, max_words_doc, multipleRefs.mydata[rti].alignedX);
						
						// vector_xi_yi indexed by rti //
					vector_xi_yi[stringify(rti)] = create_svector(xi_yi,"",1);				
					free(xi_yi);
				}
			}
			else
			{
				cerr << "Can't file  " + myParam.alignLoss + " alignLoss handler" << endl; 
				exit(-1);
			}

			// STOP HERE and THINK //
			
			// calculate number of constraints //
			if (myParam.alignLoss == "mulA")
			{
				docs = (DOC **)my_malloc(sizeof(DOC *) * nBestAnswer.size() * multipleRefs.mydata.size()); 
				rhs = (double *)my_malloc(sizeof(double) * nBestAnswer.size() * multipleRefs.mydata.size());
			}
			else
			{
				docs = (DOC **)my_malloc(sizeof(DOC *) * nBestAnswer.size());
				rhs = (double *)my_malloc(sizeof(double) * nBestAnswer.size()); // rhs constraints
			}

			// training PER //

			error += allNbestPER[0] * max(nBestAnswer[0].size(), multipleRefs.mydata[refForNBest[0]].alignedY.size());
			allPhonemeTrain += max(nBestAnswer[0].size(), multipleRefs.mydata[refForNBest[0]].alignedY.size());
			

			// create doc for updating weights //
			for (unsigned int i = 0; i < nBestAnswer.size(); i++)
			{
				SVECTOR *vector_xi_yk, *vector_diff;
				WORD *xi_yk;

				xi_yk = my_feature_map_word(myParam,featureNbest[i], nBestAnswer[i], &featureHash, max_words_doc, alignedXnBest[i]);
				vector_xi_yk = create_svector(xi_yk,"",1);
				free(xi_yk);

				if (myParam.alignLoss == "mulA")
				{
					for (unsigned int rti = 0; rti < multipleRefs.mydata.size(); rti++)
					{
						vector_diff = sub_ss(vector_xi_yi[stringify(rti)], vector_xi_yk);
						long docsID = rti+ (i * multipleRefs.mydata.size());
						docs[docsID] = create_example(docsID, docsID, docsID, 1, vector_diff);

						if (allNbestPER[i] == 0)
						{
							rhs[docsID] = 0;
						}
						else
						{
							rhs[docsID] = allNbestPER[i] - score_yi[stringify(rti)] + scoreNbest[i] + 1;
						}
					}
					free_svector(vector_xi_yk);
				}
				else
				{
					if ((myParam.alignLoss == "avgL") || (myParam.alignLoss == "rakL") || (myParam.alignLoss == "ascL"))
					{
						vector_diff = sub_ss(vector_xi_yi["avg"], vector_xi_yk);
						rhs[i] = allNbestPER[i] - score_yi["avg"] + scoreNbest[i] + 1;

					}
					else
					{
						vector_diff = sub_ss(vector_xi_yi[stringify(refForNBest[i])], vector_xi_yk);
						rhs[i] = allNbestPER[i] - score_yi[stringify(refForNBest[i])] + scoreNbest[i] + 1;
					}
					
					docs[i] = create_example(i,i,i,1,vector_diff);
					free_svector(vector_xi_yk);

					if (allNbestPER[i] == 0)
					{
						rhs[i] = 0;
					}
				}
			}

			MODEL *model = (MODEL *)my_malloc(sizeof(MODEL));

			long int totdoc;

			if (myParam.alignLoss == "mulA")
			{
				totdoc = (long int) nBestAnswer.size() * multipleRefs.mydata.size();
			}
			else
			{
				totdoc = (long int) nBestAnswer.size();
			}
			
			
			long int totwords = (long int) featureHash.size();

			LEARN_PARM learn_parm;
			KERNEL_PARM kernel_parm;
			set_default_parameters(&learn_parm, &kernel_parm);

			learn_parm.svm_c = myParam.SVMcPara;
			//learn_parm.svm_c = 9999999;
			
			KERNEL_CACHE *kernel_cache=NULL;
			double *alpha = NULL;

			svm_learn_optimization(docs, rhs, totdoc, totwords, &learn_parm, &kernel_parm, kernel_cache, model, alpha);

			// update weights w = z + o; 
			// w = new weights
			// z = update part obtained from the optimizer
			// o = old weights
			long sv_num=1;
			SVECTOR *v;
			for(long i=1;i<model->sv_num;i++) 
			{
				for(v=model->supvec[i]->fvec;v;v=v->next) 
					sv_num++;
			}

			for(long i=1;i<model->sv_num;i++)
			{
				for(v=model->supvec[i]->fvec;v;v=v->next) 
				{
					double alpha_value = model->alpha[i]*v->factor;
							
					//vector<WORD> v_words_list;
					for (long j=0; (v->words[j]).wnum; j++) 
					{
						myWF.updateFeature(my_feature_hash_retrieve(&featureHash, 
								(long)(v->words[j]).wnum),
								(double)(v->words[j]).weight * alpha_value);

					//		v_words_list.push_back(v->words[j]);
					} 
					//cout << "dummy for debug" << endl;
				}
			}

			// free memory

			for (hash_string_SVECTOR::iterator vectorPos = vector_xi_yi.begin() ; vectorPos != vector_xi_yi.end() ; vectorPos++)
			{
				free_svector(vectorPos->second);
			}
			for(long i=0;i<totdoc;i++) 
				free_example(docs[i],1);
			free(rhs);
			free_model(model,0);
			
		}// ending training all data one round

		

		//cout << "Trained on " << trainData.size() << " instances" << endl;
		cout << "Trained on " << actual_train << " instances" << endl;
		cout << "Training PER : " << (error / allPhonemeTrain) << endl;

		cout << "Error reduction on training : " << (p_error_train - error) / p_error_train << endl;

		p_error_train = error;

		cout << "Finalizing the actual/average weights .... " << endl;
		myWF.finalizeWeight(iter);
		

		if (devUnique.size() > 0)
		{
			error = 0;
			allPhonemeTrain = 0;
			actual_dev = 0;
			param myDevParam;
			myDevParam = myParam;

			myDevParam.nBest = 1;
			myDevParam.atTesting = true;

			// reset the dev check//
			//devUniqueChk.clear();

			cout << "Perform on Dev \n";

			for (hash_string_vData::iterator dev_pos = devUnique.begin(); dev_pos != devUnique.end(); dev_pos++)
			{
			//for (unsigned long di = 0; di < devData.size(); di++)
			//{
				vector_2str nBestAnswer;
				vector_2str alignedXnBest;
				vector_3str featureNbest;
				vector<double> scoreNbest;
				vector<double> nBestPER;
			
				// skip dev instance if we have already tested it in this iteration //
				/*string devWord = join(devData[di].unAlignedX,"","");
				if (devUniqueChk.find(devWord) != devUniqueChk.end())
				{
					continue;
				}
				else
				{
					devUniqueChk[devWord] = true;
				}*/

				actual_dev++;

				//nBestAnswer = phrasalDecoder(myDevParam, devData[di].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
				//nBestAnswer = phrasalDecoder(myDevParam, dev_pos->second[0].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
				
				if (myParam.useBeam)
				{
					nBestAnswer = phrasalDecoder_beam(myDevParam, dev_pos->second.mydata[0].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
				}
				else
				{
					nBestAnswer = phrasalDecoder(myDevParam, dev_pos->second.mydata[0].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
				}

				

				vector<data> multipleRefs = dev_pos->second.mydata;
				int refForTheBest;
				
				// nbi = 0; // top list when testing //
				double minPER = 9999;
				for (int rti = 0; rti < multipleRefs.size(); rti++)
				{
					double pos_PER = minEditDistance(multipleRefs[rti].alignedY, nBestAnswer[0], "_");

					if (pos_PER < minPER)
					{
						minPER = pos_PER;
						refForTheBest = rti;
					}
				}

				error += minPER * max(nBestAnswer[0].size(), multipleRefs[refForTheBest].alignedY.size());
				allPhonemeTrain += max(nBestAnswer[0].size(), multipleRefs[refForTheBest].alignedY.size());

			}

			cout << "Test on the dev set of " << actual_dev << " instances" << endl;
			cout << "Dev PER : " << (error / allPhonemeTrain) << endl;
		}

		//if (myParam.keepModel)
		//{
		cout << "Writing weights to : " << myParam.modelOutFilename << "." << iter << endl;
		myWF.writeToFile(myParam.modelOutFilename + "." + stringify(iter));
	
		cout << "Writing max phrase size to : " << myParam.modelOutFilename << "." << iter << ".maxX" << endl;
		writeMaxPhraseSize(myParam, myParam.modelOutFilename + "." + stringify(iter) + ".maxX");
	
		cout << "Writing limited phoneme/letter units to: " << myParam.modelOutFilename << "." << iter << ".limit" << endl;
		myAllPhoneme.writeToFile(myParam.modelOutFilename + "." + stringify(iter) + ".limit", true);
		//}
		

		 // clean up past model (save some space) //
		if ( (iter > 2) && (! myParam.keepModel) )
		{
			string past_modelFilename;
			past_modelFilename = myParam.modelOutFilename + "." + stringify(iter-2);
			cout << "Clean up past model : " << past_modelFilename << endl;
			if (remove(past_modelFilename.c_str()) == 0)
				cout << "Delete file " << past_modelFilename << endl;
			else
				cout << "Cannot delete file " << past_modelFilename << endl;
		}

		cout << "Error reduction ((p_error - error) / p_error) : " << ((p_error - error) / p_error) << endl;

		if ((error >= p_error) && (iter > myParam.trainAtLeast))
		{
			// stop training //
			stillTrain = false;
		}
		else
		{
			// still train //
			p_error = error;
		}

		// train at most condition
		if (iter >= myParam.trainAtMost)
		{
			stillTrain = false;
		}
	}

	// if we haven't written the model, we should write it before finishing //
	// we should write the previous model because that is what the peak is //
	
	/*if ( ! myParam.keepModel )
	{
		cout << "Writing weights to : " << myParam.modelOutFilename << "." << iter-1 << endl;
		myWF.writeToFile(myParam.modelOutFilename + "." + stringify(iter-1),true);
		
		cout << "Writing max phrase size to : " << myParam.modelOutFilename << "." << iter-1 << ".maxX" << endl;
		writeMaxPhraseSize(myParam, myParam.modelOutFilename + "." + stringify(iter-1) + ".maxX");
		
		cout << "Writing limited phoneme/letter units to: " << myParam.modelOutFilename << "." << iter-1 << ".limit" << endl;
		myAllPhoneme.writeToFile(myParam.modelOutFilename + "." + stringify(iter-1) + ".limit", true);
	}*/

	// testing if a test file is given.
	//myParam.modelInFilename = myParam.modelOutFilename + "." + stringify(iter - 1);

	// Only weight parameter we have is from the last iteration, this is probably worse than 
	// the iter - 1 on dev but who knows on the test
	// Change back since we have the previous weights stored, so test on iter-1 //
	myParam.modelInFilename = myParam.modelOutFilename + "." + stringify(iter-1);
}

void phraseModel::testing(param &myParam)
{
	myWF.clear();
	myAllPhoneme.clear(true);

	cout << endl << "Testing starts " << endl;

	cout << "Reading model file : " << myParam.modelInFilename << endl;
	myWF.updateFeatureFromFile(myParam.modelInFilename);

	cout << "Reading limited file : " << myParam.modelInFilename << ".limit" << endl;
	myAllPhoneme.addFromFile(myParam.modelInFilename + ".limit", true);

	cout << "Reading max phrase size file : " << myParam.modelInFilename << ".maxX" << endl;
	readMaxPhraseSize(myParam, myParam.modelInFilename + ".maxX");

	cout << "Max phrase size = " << myParam.maxX << endl;


	//hash_string_vData testData;
	vector_vData testData;
	cout << "Reading the test file " << endl;
	//readingAlignedFile(myParam, myParam.testingFile, testData);
	readingTestingFile(myParam, myParam.testingFile, testData);
	cout << endl << endl;

	ofstream FILEOUT, PHRASEOUT;

	if (myParam.answerFile != "")
	{		
		FILEOUT.open(myParam.answerFile.c_str(), ios_base::trunc);
		if ( ! FILEOUT)
		{
			cerr << "error: unable to create " << myParam.answerFile << endl;
			exit(-1);
		}

		string phraseOutFilename = myParam.answerFile + ".phraseOut";
		PHRASEOUT.open(phraseOutFilename.c_str(), ios_base::trunc);
		if (! PHRASEOUT)
		{
			cerr << "error: unable to create " << phraseOutFilename << endl;
			exit(-1);
		}

		cout << "Answer file : " << myParam.answerFile << endl;
		cout << "Phrase output file : " << phraseOutFilename << endl;
	}

	// at testing //
	myParam.atTesting = true;
	myParam.nBest = myParam.nBestTest;

	for (vector_vData::iterator test_pos = testData.begin(); test_pos != testData.end(); test_pos++)
	{
	/*for (unsigned long ti = 0; ti < testData.size(); ti++)
	{*/
		vector_2str nBestAnswer;
		vector_2str alignedXnBest;
		vector_3str featureNbest;
		vector<double> scoreNbest;
		vector<double> nBestPER;

		if (myParam.useBeam)
		{
			//nBestAnswer = phrasalDecoder_beam(myParam, (test_pos->second).mydata[0].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
			nBestAnswer = phrasalDecoder_beam(myParam, test_pos->mydata[0].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
		}
		else
		{
			nBestAnswer = phrasalDecoder(myParam, test_pos->mydata[0].unAlignedX, alignedXnBest, featureNbest, scoreNbest);
		}

		

		for (int n = 0; n < nBestAnswer.size(); n++)
		{
			if (myParam.answerFile != "")
			{
				FILEOUT << join(test_pos->mydata[0].unAlignedX, "", "") << "\t" << join(nBestAnswer[n], myParam.outChar, "_") << endl;
				PHRASEOUT << join(alignedXnBest[n], "|", "") << "|" << "\t";
				PHRASEOUT << join(nBestAnswer[n], "|","") << "|" << "\t";
				PHRASEOUT << n+1 << "\t";
				PHRASEOUT << scoreNbest[n] << endl;
			}
			else
			{
				cout << join(test_pos->mydata[0].unAlignedX, "", "") << "\t" << join(nBestAnswer[n], myParam.outChar, "_") << endl;
			}
		}

		if (myParam.answerFile != "")
		{
			PHRASEOUT  << endl;
		}
	}

	if (myParam.answerFile != "")
	{
		FILEOUT.close();
		PHRASEOUT.close();
	}
}

