from CONST import W2VSOURCEFILE, W2VMODEL
import word2vec
word2vec.word2vec(W2VSOURCEFILE, W2VMODEL, size=100, verbose=True)
