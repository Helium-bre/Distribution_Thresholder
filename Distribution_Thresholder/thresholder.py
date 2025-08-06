import numpy as np
from scipy import stats
from sklearn.mixture import GaussianMixture



class Thresholder :

    def __init__(self, hypothesis : str, percentage : float = None, n_components : int = 2,distributions  = ['GMM','lognormal','gamma','genextreme','beta']):
        """ Distribution based thresholder
            hypothesis : ['Anderson','Kstest'] -> The hypothesis testing method
            percentage : float -> the risk parameter, between 0 and 1. Lower percentage means more False Negative and less False Positive 
            n_components : int -> the number of components in the Gaussian Mixture distribution
            distributions : list['GMM','lognormal','gamma','genextreme','beta'] -> the list of all tested distributions"""
        
        self.hypothesis = hypothesis
        self.percentage = percentage
        self.n_components = n_components
        self.distributions = distributions
        self.fit_labels_ = None
        self.fit_survival_ = None
        self.model_ = None
        self.distributions_dict_ = {'GMM':GaussianMixture,'lognormal':stats.lognorm,'gamma':stats.gamma,\
                          'genextreme': stats.genextreme,'beta':stats.beta,'t':stats.t}
        
    
    def fit(self,data) -> None:
        """Fit a distribution to the given data
            data : array-like"""
        
        models = np.array([None for i in self.distributions])
        pvalues = np.zeros(len(self.distributions))
        data = np.asarray(data)
        for i in range(len(self.distributions)) :
            dist = self.distributions[i]
            if dist == 'GMM':
                models[i] = GaussianMixture(n_components = self.n_components).fit(data.reshape(-1,1))
                sample = models[i].sample(len(data))[0].reshape(-1)
            else :
                models[i] = self.distributions_dict_[dist](*self.distributions_dict_[dist].fit(data))
                sample = models[i].rvs(len(data))
            if self.hypothesis == 'Anderson':
                pvalues[i] = stats.anderson_ksamp([data,sample],method=stats.PermutationMethod(1000)).pvalue
            elif self.hypothesis == 'Kstest':
                pvalues[i] = stats.kstest(data,sample).pvalue

        best_index = np.argmax(pvalues)
        self.model_ = models[best_index]
        self.fit_labels_ , self.fit_survival_ = self.predict(data)

    def survival(self,x) :
        if isinstance(self.model_,GaussianMixture) :
            mcdf = np.sum([self.model_.weights_[i] * stats.norm.cdf(x,loc = self.model_.means_[i],scale = self.model_.covariances_[i]) for i in range(self.n_components)],axis = 0).flatten()

            return 1 - mcdf
        else : 
            return self.model_.sf(x)


    def predict(self,score,percentage = None) -> tuple:
        """ Predicts the label of the score based on the fitted distribution
            score : float or array-like -> The anomaly score to predict label from
            percentage (optional if it has already been entered in the object initialisation): float -> the risk parameter, between 0 and 1. Lower percentage means more False Negative and less False Positive 
            
            Output : dict ->
                - label : array-like -> labels of the input score. 0 is an inlier and 1 is an outlier
                - sf : array-like -> survival function of the input score, based on the fitted distribution
            """   
        if percentage :
            self.percentage = percentage
        elif not self.percentage : 
            raise ValueError('no risk parameter has been entered')
        if not self.model :
            raise Exception('The thresholder has not been fitted')
        
        score = np.asarray(score)
        sf = np.array([self.survival(s) for s in score]).reshape(-1)
        return {'label':sf<self.percentage,'sf':sf}
    


