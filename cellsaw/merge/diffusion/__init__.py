
import numpy as np
from ubergauss import tools
from sklearn.semi_supervised import LabelSpreading,LabelPropagation
from cellsaw.merge.diffusion.kernel import linear_assignment_kernel, linear_assignment_kernel_XXX

def stringdiffuse(mergething, labels, pid = 1,
                                sigmafac = 1,
                              neighbors_inter = 1,
                              neighbors_intra = 7,linear_assignment_factor=1):

    '''
    will use the first dataset to train and return prediction on the second
    '''

    # the data i use has theese properties:
    # -1 -> no true label
    # "Unknown" -> it was decided that we treat this as a normal label
    # "pangalo error" -> does not happen often, will be treated as normal label
    # thus encoding labels like this is fine:
    sm = tools.spacemap(
            np.unique(
                [xx for xx in labels if isinstance(xx, str)]
            ))

    diffusor = Diffusion(n_neighbors_inter = neighbors_inter,
                         sigmafac= sigmafac,
                         n_neighbors_intra  = neighbors_intra,
                         linear_assignment_factor=linear_assignment_factor)
    diffusor.fit(mergething.projections[pid][1], sm.encode(labels))
    intresults =  diffusor.predict(mergething.projections[pid][0])
    return sm.decode(intresults)

class Diffusion:

    def __init__(self,
                n_neighbors_intra = 7,
                n_neighbors_inter=1,
                sigmafac = 1,
                lp_model= LabelPropagation(kernel = None, max_iter=1000),
                linear_assignment_factor = 1,
                kernel = linear_assignment_kernel):

        """we just run diffusion as sklearn would i.e. expect no string labels etc"""

        self.neighbors_intra = n_neighbors_intra
        self.neighbors_inter = n_neighbors_inter
        self.lp_model = lp_model
        self.kernel = kernel
        self.sigmafac = sigmafac
        self.linear_assignment_factor =linear_assignment_factor


    def fit(self,X,y):
        self.X = X
        self.y = np.array(y)
        self.train_ncells = X.shape[0]
        kernel = lambda x1, x2: self.kernel(x1,x2,
                                            neighbors = self.neighbors_intra,
                                            neighbors_inter = self.neighbors_inter,
                                            sigmafac = self.sigmafac,
                                            linear_assignment_factor=self.linear_assignment_factor)
        self.lp_model.set_params(kernel = kernel)


    def predict(self,X,y = False):
        if not y:
            y = np.full(X.shape[0],-1)


        #assert self.X.shape == X.shape, 'not sure why i assert this... probably because i havent tested it'
        #assert self.y.shape == y.shape, 'assert everything!'


        # print(f"FITT")
        self.lp_model.fit(self.X, self.y)
        # print(f"{self.lp_model.label_distributions_=}")
        # print(f"PREDICT")
        res = self.lp_model.predict(X)
        return res

        # Ystack = np.hstack((self.y,y))
        # Xstack = np.vstack(Map(tools.zehidense, (self.X,X)))
        # all_labels = self.lp_model.fit(Xstack, Ystack).transduction_
        # self.correctedlabels = all_labels[self.train_ncells]
        # return  all_labels[self.train_ncells:]

