from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import logging
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import TorchMLPClassifier



#Dataset related (baddebt,cancer,iris,magic, ...)
#DATASET="baddebt"
DATASET="higgs"
#DATASET="magic"
#DATASET="cancer"
#DATASET="forest"
#DATASET="sales"
#DATASET="qsar"

DATASET_DO_SAMPLE=True
DATASET_SAMPLE_SIZE=10000
DATASET_FEATURE_SIZE=150  ##used for scalability tests only
#DATASET_SAMPLE_SIZE=int(DATASET_SAMPLE_SIZE*0.7)  ##used for METADES method only , %30 shared for TEST
#Experiments related
EXPERIMENTS_FILE="./experiments/experiments_v305_ANN_sales.xlsx"

#Model Building-feature selection related
#ALPHA=20 #used_below temporarily
#SCORER=f1_score
SCORER=accuracy_score


#Base model specific features (True) or all set of features (False)
BASE_MODEL_EXTENDED=False
#Improvement Type (Restricted or Any)
SCORE_IMPROVEMENT_TYPE="Any"

#Maturity Scorer Improvement Type (Restricted or Any)
MATURITY_SCORER="in"

#MODEL_LIST_STRING="XGBClassifier,RandomForestClassifier,KNeighborsClassifier,GaussianNB,DecisionTreeClassifier,LGBMClassifier"
#MODEL_LIST_STRING="XGBClassifier"
#MODEL_LIST_STRING="XGBClassifier,RandomForestClassifier,KNeighborsClassifier,GaussianNB"
#MODEL_LIST_STRING="XGBClassifier,TorchMLPClassifier"
#MODEL_LIST_STRING="RandomForestClassifier,DecisionTreeClassifier"
MODEL_LIST_STRING="RandomForestClassifier,KNeighborsClassifier,XGBClassifier,DecisionTreeClassifier,LGBMClassifier,GaussianNB"
#MODEL_LIST_STRING="RandomForestClassifier,KNeighborsClassifier,XGBClassifier,DecisionTreeClassifier,LGBMClassifier,GaussianNB,TorchMLPClassifier"
#MODEL_LIST_STRING="RandomForestClassifier,XGBClassifier,DecisionTreeClassifier,KNeighborsClassifier,GaussianNB,LGBMClassifier"
#MODEL_LIST_STRING="RandomForestClassifier"
#MODEL_LIST_STRING="TorchMLPClassifier"

def create_model_list(models):
    base_models=[]
    for model in models.split(","):
        if(model=="RandomForestClassifier"):
            base_models.append(RandomForestClassifier())
            #base_models.append(RandomForestClassifier(random_state=12))
            #base_models.append(RandomForestClassifier(random_state=13))
            #base_models.append(RandomForestClassifier(random_state=14))
        if(model == "DecisionTreeClassifier"):
            base_models.append(DecisionTreeClassifier())
            #base_models.append(DecisionTreeClassifier(random_state=100))
            #base_models.append(DecisionTreeClassifier(random_state=101))
            #base_models.append(DecisionTreeClassifier(random_state=102))

        if (model == "KNeighborsClassifier"): base_models.append(KNeighborsClassifier())
        if (model == "GaussianNB"):
            base_models.append(GaussianNB())
            #base_models.append(GaussianNB(var_smoothing=0.0000000002))
            #base_models.append(GaussianNB(var_smoothing=0.0000000003))
            #base_models.append(GaussianNB(var_smoothing=0.0000000004))


        if (model == "LGBMClassifier"):
            base_models.append(LGBMClassifier(verbose=-1))
            #base_models.append(LGBMClassifier(verbose=-1,random_state=100))
            #base_models.append(LGBMClassifier(verbose=-1, random_state=101))
            #base_models.append(LGBMClassifier(verbose=-1, random_state=102))

        if (model == "XGBClassifier"):
            base_models.append(XGBClassifier(verbosity=0, silent=True))
            #base_models.append(XGBClassifier(verbosity=0, silent=True,random=101))
            #base_models.append(XGBClassifier(verbosity=0, silent=True,random=102))
            #base_models.append(XGBClassifier(verbosity=0, silent=True, random=103))

        if (model == "TorchMLPClassifier"):
            #base_models.append(TorchMLPClassifier.build_mlp_classifier(input_size=10, hidden_sizes=[64, 32], output_size=2, epochs=20)) ###MAGIC
            #base_models.append(TorchMLPClassifier.build_mlp_classifier(input_size=32, hidden_sizes=[64, 32], output_size=2, epochs=20)) ###HIGGS
            #base_models.append(TorchMLPClassifier.build_mlp_classifier(input_size=54, hidden_sizes=[64, 32], output_size=7,epochs=20))  ###FOREST
            #base_models.append(TorchMLPClassifier.build_mlp_classifier(input_size=99, hidden_sizes=[64, 32], output_size=2,epochs=20))  ###BADDEBT
            base_models.append(TorchMLPClassifier.build_mlp_classifier(input_size=4, hidden_sizes=[64, 32], output_size=2,epochs=20))  ###SALES

    return base_models

MODEL_LIST=create_model_list(MODEL_LIST_STRING)

EXPERIMENT_FEATURES=False #True if enable testing i.e. StackingClassifier with reduced (previously found) feature set
###HIGGS #28 features
EXPERIMENTAL_FEATURE_LIST_HIGGS=['EventId', 'DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_num', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_subleading_phi', 'PRI_jet_all_pt', 'Weight']
EXPERIMENTAL_FEATURE_LIST_HIGGS_rfe_20=['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_leading_eta']
EXPERIMENTAL_FEATURE_LIST_HIGGS_rfe_24=['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi']
EXPERIMENTAL_FEATURE_LIST_HIGGS_rfe_28=['DER_mass_MMC', 'DER_mass_transverse_met_lep', 'DER_mass_vis', 'DER_pt_h', 'DER_deltaeta_jet_jet', 'DER_mass_jet_jet', 'DER_prodeta_jet_jet', 'DER_deltar_tau_lep', 'DER_pt_tot', 'DER_sum_pt', 'DER_pt_ratio_lep_tau', 'DER_met_phi_centrality', 'DER_lep_eta_centrality', 'PRI_tau_pt', 'PRI_tau_eta', 'PRI_tau_phi', 'PRI_lep_pt', 'PRI_lep_eta', 'PRI_lep_phi', 'PRI_met', 'PRI_met_phi', 'PRI_met_sumet', 'PRI_jet_leading_pt', 'PRI_jet_leading_eta', 'PRI_jet_leading_phi', 'PRI_jet_subleading_pt', 'PRI_jet_subleading_eta', 'PRI_jet_all_pt']



###FOREST
EXPERIMENTAL_FEATURE_LIST_FOREST_32_rfe= ['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points', 'Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3', 'Wilderness_Area4', 'Soil_Type2', 'Soil_Type4', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12', 'Soil_Type13', 'Soil_Type20', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24', 'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
EXPERIMENTAL_FEATURE_LIST_FOREST_32_alpha=['Hillshade_9am', 'Soil_Type11', 'Slope', 'Hillshade_3pm', 'Soil_Type4', 'Soil_Type33', 'Wilderness_Area3', 'Soil_Type20', 'Wilderness_Area2', 'Vertical_Distance_To_Hydrology', 'Soil_Type22', 'Soil_Type24', 'Horizontal_Distance_To_Hydrology', 'Soil_Type10', 'Wilderness_Area4', 'Soil_Type23', 'Soil_Type29', 'Soil_Type32', 'Aspect', 'Soil_Type2', 'Soil_Type39', 'Soil_Type13', 'Soil_Type16', 'Horizontal_Distance_To_Fire_Points', 'Hillshade_Noon', 'Horizontal_Distance_To_Roadways', 'Soil_Type12', 'Soil_Type27', 'Elevation', 'Soil_Type31', 'Soil_Type30', 'Wilderness_Area1']


EXPERIMENTAL_FEATURE_LIST=EXPERIMENTAL_FEATURE_LIST_HIGGS_rfe_28
#EXPERIMENTAL_FEATURE_LIST=EXPERIMENTAL_FEATURE_LIST_HIGGS_20
#EXPERIMENTAL_FEATURE_LIST=EXPERIMENTAL_FEATURE_LIST_HIGGS_28
ALPHA=18 #pick all

DATA_ANN_COLUMN_COUNT = 0
DATA_ANN_TARGET_VALUE_COUNT = 0


#Margin and Diversitiy ordered Ensemble sorted models
MDM_MODEL_LIST=[]

#Feature selection related
SHAP_SAMPLE_SIZE=50
#SHAP_SAMPLE_SIZE=200
#LOAD_SHAP_FEATURES=True
LOAD_SHAP_FEATURES=True
SHAP_LIST_FOLDER="./resources/shap_list"

def print_global_parameters():
    logger = logging.getLogger()
    logger.info("ALPHA="+str(ALPHA))
    logger.info("SCORER=" + str(SCORER))
    logger.info("DATASET=" + str(DATASET))
    logger.info("DO_SAMPLE=" + str(DATASET_DO_SAMPLE))
    logger.info("SHAP_SAMPLE_SIZE=" + str(SHAP_SAMPLE_SIZE))
    logger.info("LOAD_SHAP_FEATURES=" + str(LOAD_SHAP_FEATURES))
    logger.info("SHAP_LIST_FOLDER=" + str(SHAP_LIST_FOLDER))

def get_global_parameters():
    params = {"DATASET": DATASET,
              "DATASET_DO_SAMPLE": DATASET_DO_SAMPLE,
              "SCORER": SCORER,
              "ALPHA": ALPHA,
              "SHAP_SAMPLE_SIZE": SHAP_SAMPLE_SIZE,
              "MODEL_LIST": MODEL_LIST,
              "BASE_MODEL_EXTENDED": BASE_MODEL_EXTENDED,
              "SCORE_IMPROVEMENT_TYPE": SCORE_IMPROVEMENT_TYPE
              }
    return params



