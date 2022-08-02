import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster import hierarchy as hira
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import precision_score
import pandas as pd

def dendro(similarity, labels, distancecut = 1):

    # HEATMAP
    f=plt.figure(figsize=(16,8))
    ax=plt.subplot(121)
    ax.set_title('complete similarity matrix', fontsize=20)
    sns.heatmap(similarity, xticklabels = False,
            yticklabels = labels,  cmap="YlGnBu")
    locs, labels = plt.yticks()
    plt.setp(labels,size = 8)

    # DENDROGRAM
    ax=plt.subplot(122)
    ax.set_title('induced dendrogram (ward)', fontsize=20)
    Z = squareform(similarity)
    Z = hira.linkage(1-Z,'ward')
    hira.dendrogram(Z, labels = labels, color_threshold=distancecut, orientation='right')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90,size = 8)
    plt.subplots_adjust(wspace = .4)
    return hira.fcluster(Z, t=7, criterion = 'maxclust')



def plotPrecision(data, wg = True, method='cosine similarity'):
    # plor precision at k :)
    # [val,rep,p@k,n_genes]
    df = pd.DataFrame(data)
    df.columns = 'precision rep neighbors±σ genes '.split()
    title = f'Searching for similar datasets via {method}'

    if wg:
        sns.set_theme(style='whitegrid')
    sns.lineplot(data = df,x='genes', y = 'precision',style= 'neighbors±σ',
            hue = 'neighbors±σ',palette="flare", ci=68)
    plt.title(title, y=1.06, fontsize = 16)
    plt.ylim([.75,1])
    plt.ylabel('precision of neighbors (40 datasets)')
    plt.xlabel('number of genes')




