from lmz import Map,Zip,Filter,Grouper,Range,Transpose,Flatten
import pprint
import scalp.data.align as data
from collections import Counter, defaultdict
from ubergauss.tools import spacemap
import numpy as np

import plotly.io as pio
import io
from PIL import Image

import matplotlib

from matplotlib import pyplot as plt



def mkcolors(label):
    colorsm = spacemap(np.unique(label))
    cmap = plt.cm.get_cmap('turbo', len(colorsm.integerlist))
    myrgb = Map(cmap, colorsm.encode(label))
    return Map(matplotlib.colors.rgb2hex, myrgb)


def  add_by_leftk(cnt, leftk, support_ab, support_ba):
    a_outcount = defaultdict(list)
    for (a,b),count in cnt.items():
        a_outcount[a].append((count,a,b))

    clean_count = {}
    for a in a_outcount.keys():
        asd = sorted(a_outcount[a], key = lambda x: x[0], reverse= True)[:leftk]
        for count, a, b in asd:
            clean_count[(a,b)] = count
            support_ba.pop(b, None)
            support_ab.pop(a, None)
    return clean_count


def add_by_rightk(cnt, rightk, support_ab, support_ba):

    b_outcount = defaultdict(list)
    for (a,b),count in cnt.items():
        b_outcount[b].append((count,a,b))

    clean_count = {}
    for bkey in b_outcount.keys():
        asd = sorted(b_outcount[bkey], key = lambda x: x[0], reverse= True)[:rightk]
        for count, a, b in asd:
            clean_count[(a,b)] = count
            support_ba.pop(b, None)
            support_ab.pop(a, None)
    return clean_count


def  add_by_thresh(cnt, thresh, support_ab, support_ba):
    # sum outgoing for each source
    a_outcount = defaultdict(int)
    for (a,b),count in cnt.items():
        a_outcount[a] += count

    clean_count = {}
    # add what is passing the threshold
    for (a,b),count in cnt.items():
        if count > a_outcount[a] * thresh:
            clean_count[(a,b)] = count
            # popping b so we will be left with the unaccounted
            support_ba.pop(b, None)
            support_ab.pop(a, None)

    return clean_count


def clean_counter(cnt, thresh=1, leftk = 0, rightk = 0):
    '''
    we remove connectins from the counter to remove noise
    - threshold is for the outflow of a..
        if a connection a->b has below threshold connections, we remove it
        however we can not remove all such connections as lone
        instances in b need to be preserverd

    leftK(rightK): consider connections between two batches, an intem at the left(right)
                   can have <= k , outgoin connections (thicker connections prefered)
    '''
    # we cant drop targets so we keep a list and remove the covered ones later
    support_ba = defaultdict(list)
    for (a,b),count in cnt.items():
        support_ba[b].append([count, a])

    support_ab = defaultdict(list)
    for (a,b),count in cnt.items():
        support_ab[a].append([count, b])


    clean_count = add_by_thresh(cnt, thresh, support_ab, support_ba)
    clean_count.update(add_by_leftk(cnt, leftk, support_ab, support_ba))
    clean_count.update(add_by_rightk(cnt, rightk, support_ab, support_ba))


    # add the remaining b instances
    # adding all adds too much junk
    # so we sort and only add one
    for b,aa in support_ba.items():
        aa = sorted(aa, key = lambda x:x[0], reverse = True)
        a = aa[0][1]
        clean_count[(a,b)] = cnt[(a,b)]
    for a,bb in support_ab.items():
        bb = sorted(bb, key = lambda x:x[0], reverse = True)
        b = bb[0][1]
        clean_count[(a,b)] = cnt[(a,b)]


    return clean_count


def adatas_to_sankey(adatas, thresh = .1, leftk = 0, rightk = 0, labelfield = f'label'):
    source,target ,value = [],[],[]

    node_groups = []

    for i in range(len(adatas)-1):
        a1 = adatas[i]
        a2 = adatas[i+1]
        c = Counter(zip(a1.obs[labelfield],a2.obs[labelfield]))
        c = clean_counter(c, thresh = thresh, leftk=leftk, rightk= rightk)

        s,t = Transpose(list(c.keys()))
        source+=[ss+str(i) for ss in s]
        target+=[tt+str(i+1) for tt in t]

        value += list(c.values())
        node_groups.append( [ss+str(i) for ss in s])


    sm = spacemap(np.unique(source+target))

    label = [s[:-1] for s in sm.itemlist ]

    # node_groups = Map(sm.encode, node_groups) doesnt work //

    return {'label':label, 'color':mkcolors(label)}, {'source':sm.encode(source), 'target':sm.encode(target), 'value':value}


def adatas_to_sankey_fig(adatas, align = False, thresh = .15,
                         leftk= 0, rightk = 0, label ='label', title = ''):
    import plotly.graph_objects as go
    if align:
        data.align(adatas, base = align)

    node,link = adatas_to_sankey(adatas, thresh = thresh, leftk=leftk,
                                 rightk= rightk, labelfield = label)
    fig = go.Figure(data=[go.Sankey(
        node = node,
        link = link
        )])

    fig.update_layout( hovermode = 'x', title=title,
    font=dict(size = 10, color = 'black'),
    plot_bgcolor='white',
    paper_bgcolor='white')

    return fig

def plt_plotly(plotly_fig):
    image_bytes = pio.to_image(plotly_fig, format='png')
    pil_image = Image.open(io.BytesIO(image_bytes))
    plt.figure()
    plt.imshow(pil_image)
    plt.axis('off')
    plt.show()

def plot(dataset, **kwargs):
    fig = adatas_to_sankey_fig(dataset, **kwargs)
    plt_plotly(fig)



def test_sankey():
    from scalp import data, pca, diffuse, umapwrap, mnn, graph, test_config
    from scalp.output import score
    a = data.loaddata_scib(test_config.scib_datapath,
                           maxdatasets=3,
                           maxcells = 600,
                           datasets = ["Immune_ALL_hum_mou.h5ad"])[0]


    matplotlib.use('module://matplotlib-sixel')
    plotly_fig = adatas_to_sankey_fig(a, thresh = .15,align="X",leftk= 0, rightk = 0, label ='label')
    plt_plotly(plotly_fig)
    print()

    a = diffuse.diffuse_label_sklearn(a, ids_to_mask=[2,1], new_label ='skdiff')
    print(f"{Map(score.anndata_ari, a, predicted_label='skdiff')=}")



