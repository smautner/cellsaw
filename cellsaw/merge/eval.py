
from sklearn.metrics import adjusted_rand_score
import numpy as np
def evaluate(pairs, labels, setid = 1):
    scores = np.array([scorepair(p,l,setid = setid) for p,l in zip(pairs, labels)])
    return scores

def scorepair(pair, labels, setid=1):
    return adjusted_rand_score( pair.data[setid].obs['true'],labels[setid])
