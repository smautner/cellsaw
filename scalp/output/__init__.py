from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
from scalp.output.sankey import plot
from scalp.output.score import anndata_ari

def sankeyscore(dataset,cmp_label='compare label to this label',**kwargs):

    plot(dataset,**kwargs)
    rand_scores_per_batch = Map(anndata_ari, dataset, predicted_label=cmp_label)
    print(f"{rand_scores_per_batch= }")


