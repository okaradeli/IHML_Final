import logging
import utils.global_init as gi #Logging module initialized first
import globals as glb

from numpy import mean
from numpy import std

#Custom modules
import model_loader as ml
import data_loader as dl
import featureselector as fs
import base_model as bm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, recall_score,precision_score,roc_auc_score
from datetime import datetime
import time
import pandas as pd


##This is PHD project for Onur Karadeli

def get_learner_names(M):
    str_learner_names=""
    for model in M:
        name = model.base_model.__class__.__name__
        str_learner_names+=name+","
    return str_learner_names


class IncrementalMetalearner(ClassifierMixin):

    logger=None
    neptune_run=None
    experiment_name= None
    M={} #Best Base learner list
    F={} #Best features list
    model=None #The trained model if any

    def __init__(self):
        self.logger = logging.getLogger()
        #self.set_up_neptune()

    def fit(self, X, Y,dataset=glb.DATASET):
        return self._fit(X,Y,dataset)

    def predict(self,X):
        X=X[self.F]
        Y_pred = ml.predict_incremental_model(self.model, X )
        print("preds complete. Size:"+str(len(Y_pred)))
        return Y_pred
    def predict_proba(self,X):
        X = X[self.F]
        Y_pred = self.model.predict_proba(X)
        return Y_pred

    def _fit(self, X,Y,dataset):
        # get the models to evaluate
        models = ml.get_base_models()

        train_start_time = time.time()

        #Incremental MetaLearner
        M=list()
        #Incremental FeatureSet
        F=list()
        #Initial base model Accuracy
        H_base_models=list()
        #Top Features
        F_dict=dict()
        #Init variables
        score_current=0

        #get top alpha features
        for model in models:
            name = model.__class__.__name__
            model_top_features = fs.get_top_features(model, glb.ALPHA, X, Y,dataset)
            self.logger.info("Processing model:"+name)
            F_dict[model]=model_top_features

        # Evaluate the models and store initial results
        results, names = list(), list()
        for model in models:
            scores = ml.evaluate_model_train(model, X, Y)
            ##scores = ml.evaluate_model_train(model, X[F_dict[model]], Y) ##fixed a bug , ALL features was being input to model train , but it should be F instead.
            score=mean(scores)

            base_model_obj=bm.BaseModel(model,F_dict[model],glb.ALPHA)
            H_base_models.append((base_model_obj, score))

            results.append(scores)
            name=model.__class__.__name__
            names.append(name)
            #logger.info('>%s %.3f (%.3f)' % (name, mean(scores), std(scores)))

        #Sort models on accuracy
        ml.sort_models(H_base_models)

        #Incremental Learner building
        for model,score in H_base_models:
            self.logger.info("Evaluating Model:" + str(ml.get_model_name(model.base_model)) )
            #Empty Metalearner list (i.e. initialization) add first learner
            if not M:
                M.append(model)
                H_new_features = F_dict[model.base_model]
                F = fs.append_features([], H_new_features)
                self.logger.info("FIRST BEST Model:" + str(ml.get_model_name(model.base_model)) )##+ " score:" + str(score))
                recalculate_current_score=True
                continue

            #Build new metalearner and new featureset
            M_new=list(M)
            M_new.append(model)
            H_new_features=F_dict[model.base_model]
            F_new=fs.append_features(F,H_new_features)

            #Get current and new metalearner scores
            #if there is no change (no new feature, no new base leaerner) dont re-evaluate the model
            if recalculate_current_score:
                self.logger.info("Change detected, re evaluating current score.")
                score_current = ml.train_evaluate_incremental_model(M, F, X, Y) ##only if something changed
            score_new =     ml.train_evaluate_incremental_model(M_new, F_new, X, Y)

            #Compare results to find out improvement
            score_improving= (score_new >= score_current)

            # Either Iteration validation score is improving Margins & Diversity is increasing
            if (score_improving):
                #There is improvement so update metalearner and featuresubset
                self.logger.info("IMPROVEMENT ! Adding new base learner.")
                M=M_new
                F=F_new
                self.logger.info("Score: "+str(score_new)+" Model count:"+str(len(M_new))+" feature count:"+str(len(F_new))+" new model:"+str(ml.get_model_name(model.base_model)))
                recalculate_current_score=True
            else:
                self.logger.info("NO IMPROVEMENT. Skipping, skipped model:"+str(ml.get_model_name(model.base_model)))
                recalculate_current_score = False

        #Best baselearner list
        self.logger.info("Best base learner list:"+str(len(M)))
        ml.print_model_list(M)
        self.logger.info("Best features list:"+str(len(F)))
        self.logger.info(F)
        self.logger.info("-----")
        self.M=M
        self.F=F
        self.model=ml.train_incremental_model(M,F,X,Y)

        train_end_time = time.time() ##total train time
        return str(round(train_end_time - train_start_time, 4))


    def evaluate_experiment(self, X=None, Y=None):
        experiement_outputs = {}


        #Final Evaluation
        ##Single Models and StackedGeneralization
        models=ml.get_base_models()
        models.append(ml.get_stacked_model(models))

        ##For test purposes add MLEns-StackedGeneratlization
        ##MLENS
        from mlens.ensemble import BlendEnsemble
        from sklearn.linear_model import LogisticRegression
        """Return an ensemble."""
        ensemble = BlendEnsemble()
        ensemble.add(models, proba=False)  # Specify 'proba' here
        ensemble.add_meta(LogisticRegression())
        ##Add MLEns to the final evaluation list
        models.append(ensemble)

        #experiement_outputs["IncrementalStackedGeneraliztion"] = mean(scores)
        # Evaluate the models and store initial results
        results, names = list(), list()

        for model in models:
            if model.__class__.__name__=='BlendEnsemble'  and glb.SCORER==roc_auc_score:
                continue ##for BlendEnsemble ( MLEnsemble) roc_auc_score does not work

            experiement_start_time = time.time()##experiement_start_time = timeit.timeit()
            scores = ml.evaluate_model_train(model, X, Y)
            results.append(scores)
            name=model.__class__.__name__
            names.append(name)

            experiement_end_time = time.time()
            self.logger.info('>Test Score:%s %.3f (%.3f) TIME sec: (%.2f)' % (name, mean(scores), std(scores),round(experiement_end_time-experiement_start_time,4)))

            experiement_outputs[name]=str(round(mean(scores),4))+"/ TIME sec:"+str(round(experiement_end_time-experiement_start_time,4))
            experiement_outputs["DATASET_SIZE"] = len(X)


        ##Incremental Stack Generalization
        experiement_start_time = time.time()

        valid_features = [c for c in self.F if c in X.columns]
        #scores = ml.evaluate_model_train(self.model, X[self.F], Y)##BUG FIX here, previously it was X instead of X[self.F]
        scores = ml.evaluate_model_train(self.model, X[valid_features], Y)##BUG FIX here, previously it was X instead of X[self.F]

        experiement_end_time = time.time()
        self.logger.info('>Test Score:%s %.3f (%.3f) TIME sec:(%.2f)' % ("IncrementalStackedGeneraliztion", mean(scores), std(scores),round(experiement_end_time-experiement_start_time,4)))
        experiement_outputs["IncrementalStackedGeneraliztion"]=str(round(mean(scores),4))+"/ TIME sec:"+str(round(experiement_end_time-experiement_start_time,4))

        ##IHML
        experiement_outputs["IHMLBestFeatureSet"] = str(self.F).replace("[","").replace("]","")
        experiement_outputs["IHMLBestLearnerSet"] = get_learner_names(self.M)



        #self.neptune_run["IncrementalStackedGeneraliztion"] = score
        #self.neptune_run.stop()
        self.logger.info("New Experiment is complete.")

        return  experiement_outputs

    def run_experiment(self,X=None,Y=None):

        self.logger.info("New Experiment is running...")
        glb.print_global_parameters()

        # load dataset
        if glb.DATASET == "cancer":
            X, Y,target_column = dl.get_cancer_dataset()
        elif glb.DATASET == "iris":
            X, Y,target_column = dl.get_iris_dataset()
        elif glb.DATASET == "baddebt":
            X, Y,target_column = dl.get_baddebit_dataset()
        elif glb.DATASET == "magic":
            X, Y,target_column = dl.get_magic_dataset()
        elif glb.DATASET == "higgs":
            X, Y,target_column = dl.get_higgs_boson_dataset()
        elif glb.DATASET == "forest":
            X, Y,target_column = dl.get_forest_dataset()
        elif glb.DATASET == "sales":
            X, Y,target_column = dl.get_sales_dataset()
        elif glb.DATASET == "qsar":
            X, Y,target_column = dl.get_qsar_dataset()

        else:
            self.logger.info("INVALID DATASET, EXITING !!!")
            exit(-1)


        XTrain, XTest, YTrain, YTest = train_test_split(X, Y, test_size=0.30, random_state=42)
        params = {'verbose': -1}
        total_train_time = self.fit(XTrain,YTrain,glb.DATASET)
        experiment_outputs= self.evaluate_experiment(XTest,YTest)
        experiment_outputs["IHML_Train_Time"] = total_train_time

        ##Calculate AUC/ROC...
        ##ml.evaluate_model_roc(self,XTest,YTest)

        return experiment_outputs




    def set_params(self,params):
        #Changeable params
        if "DATASET" in params: glb.DATASET = params["DATASET"]
        if "ALPHA" in params: glb.ALPHA = params["ALPHA"]
        if "SCORER" in params:
            if params["SCORER"] == "accuracy":glb.SCORER=accuracy_score
            if params["SCORER"] == "f1": glb.SCORER = f1_score
            if params["SCORER"] == "recall":glb.SCORER=recall_score
            if params["SCORER"] == "precision": glb.SCORER = precision_score
            if params["SCORER"] == "roc_auc_score": glb.SCORER = roc_auc_score
        if "DATASET_SAMPLE_SIZE" in params: glb.DATASET_SAMPLE_SIZE = params["DATASET_SAMPLE_SIZE"]
        if "DATASET_FEATURE_SIZE" in params: glb.DATASET_FEATURE_SIZE = params["DATASET_FEATURE_SIZE"]
        if "MODEL_LIST" in params and params["MODEL_LIST"] != None:
            glb.MODEL_LIST=glb.create_model_list(params["MODEL_LIST"])

        print("set experiment params complete")



    def set_up_neptune(self):
        ##Use NEPTUNE for ML experiement logging
        #self.neptune_run = neptune.init(
        #    project="okaradeli/incrmetalearner",
        #    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIzYTQzODcyOC01MDNlLTRlMWYtYWI1Ni04N2UyOTcxOGQzZjUifQ==",
        #)  # your credentials
        params = glb.get_global_parameters()
        self.neptune_run["parameters"] = params

    def calculate_p_values(self,target_column,data):
        import statsmodels.api as sm
        from sklearn.model_selection import train_test_split
        # If p-value < 0.05 -->Significant
        # If p-value > 0.05 -->Not Significant
        prices = data[target_column]
        features = data.drop(target_column, axis=1)
        X_train, X_test, Y_train, Y_test = train_test_split(features, prices, test_size=.2, random_state=10)
        x_incl_cons = sm.add_constant(X_train)
        model = sm.OLS(Y_train, x_incl_cons)  # ordinary least square
        results = model.fit()  # regresssion results
        # results.params
        # results.pvalues
        pvalues = pd.DataFrame({'coef': results.params, 'pvalue': round(results.pvalues, 3)})
        self.logger.info(pvalues)

    def calculate_model_scaleability(self):
        return 1






#logger = logging.getLogger()
#logger.info("*****")
#logger.info("DEV started.")
#logger.info("*****")

def experiment_model_scalability():
    global logger, model
    ##EXPERIMENT MODEL SCALABILITY
    logger = logging.getLogger()
    from xgboost import XGBClassifier
    scalability_test_results = {}
    for x in range(1, 20):
        glb.MODEL_LIST = set()
        ##add models to model list
        for y in range(x):
            model = XGBClassifier(verbosity=0, silent=True)
            glb.MODEL_LIST.add(model)
        # execute scalability experiment
        experiement_start_time = time.time()
        experiment.run_experiment()
        experiement_end_time = time.time()
        scalability_test_results[x] = int(experiement_end_time) - int(experiement_start_time)
        logger.info("SCALABILITY TEST COMPLETE number of models:" + str(
            len(glb.MODEL_LIST)) + " for dataset" + glb.DATASET + " is:" + str(
            int(experiement_end_time - experiement_start_time)))
    logger.info("SCALABILITY TEST RESULTS")
    logger.info(scalability_test_results)




if __name__ == '__main__':
    experiment = IncrementalMetalearner()
    experiment.run_experiment()
    #experiment_model_scalability() Test model scalability i.e. 1..20 models time cost





