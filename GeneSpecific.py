import matplotlib
matplotlib.use('Agg')


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


dfx = pd.read_csv('df5.tsv', sep="\t")
dfx.replace('.', np.nan, inplace=True)
import sys
gene=sys.argv[1]
outfile=sys.argv[2]

df=dfx[dfx['gene']==gene]


basics=['#chr','pos','ref','alt','gene','value','confidence', 'WhatSet']
caddn=[ 'ConsScore',  'GC', 'CpG', 'mapAbility20bp', 'mapAbility35bp', 'scoreSegDup', 'priPhCons', 'mamPhCons', 'verPhCons', 'priPhyloP', 'mamPhyloP', 'verPhyloP', 'GerpN', 'GerpS', 'GerpRS', 'GerpRSpval', 'bStatistic', 'mutIndex', 'dnaHelT', 'dnaMGW', 'dnaProT', 'dnaRoll','fitCons', 'cHmmTssA', 'cHmmTssAFlnk', 'cHmmTxFlnk', 'cHmmTx', 'cHmmTxWk', 'cHmmEnhG', 'cHmmEnh',  'cHmmHet', 'cHmmTssBiv', 'cHmmBivFlnk', 'cHmmEnhBiv', 'cHmmReprPC', 'cHmmReprPCWk', 'cHmmQuies', 'EncExp', 'EncH3K27Ac', 'EncH3K4Me1', 'EncH3K4Me3', 'EncNucleo', 'EncOCC', 'EncOCCombPVal', 'EncOCDNasePVal', 'EncOCFairePVal', 'EncOCpolIIPVal', 'EncOCctcfPVal', 'EncOCmycPVal', 'EncOCDNaseSig', 'EncOCFaireSig', 'EncOCpolIISig', 'EncOCctcfSig', 'EncOCmycSig',  'motifEScoreChng', 'minDistTSS', 'minDistTSE','relcDNApos', 'CDSpos', 'relCDSpos', 'PHRED']
caddc=['AnnoType','Consequence','ConsDetail',  'Segway', 'tOverlapMotifs',  'motifEName','isKnownVariant', 'Dst2SplType']
caddn0=['motifECount','TFBS',  'TFBSPeaks',  'TFBSPeaksMax', 'TG_AF']
caddabs=[ 'motifDist','Dst2Splice']
notsure=['mirSVR-Score', 'mirSVR-E', 'mirSVR-Aln', 'targetScan' ]

defc=['RegulomeDB_score','network_hub','sensitive','target_gene','fathmm-MKL_non-coding_pred','fathmm-MKL_coding_group','hESC_Topological_Domain','IMR90_Topological_Domain','ENCODE_TFBS','ENCODE_Dnase_score','Ensembl_Regulatory_Build_feature_type','Ensembl_Regulatory_Build_TFBS']
cellt=['Ensembl_HeLa_S3_activity','Ensembl_HELAS3_segmentation','ENCODE_Helas3_segmentation','Ensembl_K562_activity','Ensembl_K562_segmentation','ENCODE_K562_segmentation','Ensembl_HepG2_activity','Ensembl_HEPG2_segmentation','ENCODE_Hepg2_segmentation']
defn=['MAP20','MAP35','MAP20(+-149bp)','MAP35(+-149bp)','GMS_single-end','GMS_paired-end','phastCons100way_vertebrate_rankscore','GERP_RS_rankscore','integrated_fitCons_rankscore','GM12878_fitCons_rankscore','H1-hESC_fitCons_rankscore','HUVEC_fitCons_rankscore','GenoCanyon_rankscore','funseq_noncoding_score','funseq2_noncoding_rankscore','CADD_phred','DANN_rank_score','fathmm-MKL_non-coding_rankscore','Eigen-PC-phred']
defn0=['splicing_consensus_ada_score','splicing_consensus_rf_score','1000Gp3_AC','UK10K_AC','TWINSUK_AC','gnomAD_genomes_AC']
maybec=['ANNOVAR_ensembl_Effect','SnpEff_ensembl_Effect','SnpEff_ensembl_Effect_impact','SnpEff_ensembl_Distance_to_feature','SnpEff_ensembl_TF_binding_effect','SnpEff_ensembl_TF_name','SnpEff_ensembl_summary','RegulomeDB_motif','Motif_breaking','ENCODE_annotated','FANTOM5_enhancer_differentially_expressed_tissue_cell']

o1=['ENCODE_TFBS_score']
possc=['ORegAnno_type', 'FANTOM5_CAGE_peak_permissive']

mc=['meanDNase.macs2', 'meanH2A.Z', 'meanH2AK5ac', 'meanH2AK9ac', 'meanH2BK120ac', 'meanH2BK12ac', 'meanH2BK15ac', 'meanH2BK20ac', 'meanH2BK5ac', 'meanH3K14ac', 'meanH3K18ac', 'meanH3K23ac', 'meanH3K23me2', 'meanH3K27ac', 'meanH3K27me3', 'meanH3K36me3', 'meanH3K4ac', 'meanH3K4me1', 'meanH3K4me2', 'meanH3K4me3', 'meanH3K56ac', 'meanH3K79me1', 'meanH3K79me2', 'meanH3K9ac', 'meanH3K9me1', 'meanH3K9me3', 'meanH3T11ph', 'meanH4K12ac', 'meanH4K20me1', 'meanH4K5ac', 'meanH4K8ac', 'meanH4K91ac', 'meanDNase.hotspot.all', 'meanDNase.hotspot.fdr0.01', 'meanRoadmap']

aacols=['valuec', 'Trimer', 'TrimerAvg', 'TrimerMut','Tetra']


for i in caddn0:
    df[i]=df[i].fillna(0)
for j in defn0:
    df[j]=df[j].fillna(0)

df['multic']=df.apply(lambda x: -1.0 if (x['value'] < -0.1) else 0 if (x['value'] >= -0.1 and x['value'] < 0.1) else 1.0, axis=1)

numeric_feats=caddn+caddn0+caddabs+defn+defn0+['TrimerAvg']
cat_feats=caddc+defc+['Trimer','TrimerMut']

df[numeric_feats]=df[numeric_feats].convert_objects(convert_numeric=True)

from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute
X = KNN(k=3).complete(df[numeric_feats])


X=df[numeric_feats+cat_feats]
import category_encoders as ce
encoder = ce.OneHotEncoder(cols=cat_feats)
X = encoder.fit_transform(X)

X2=X.shift(1)
X2.columns = [str(col) + '_2' for col in X2.columns]
X=pd.concat([X,X2],axis=1).fillna(0)

X2=pd.DataFrame(data=X, columns=df[numeric_feats].columns, index=df[numeric_feats].index)
labelX=['WhatSet','#chr','pos', 'ref','alt','gene']
vals=['value','valuec','multic', 'confidence']
X3=X2.join(df[labelX+vals])

trainset=X3[X3['WhatSet']=="Train"]
X4train=trainset[numeric_feats]
testset=X3[X3['WhatSet']=="Test"]
X4test=testset[numeric_feats]
y=trainset['value']
yc=trainset['valuec']
ym=trainset['multic']
conf=trainset['confidence']

import xgboost as xgb
import lightgbm as lgb
from skopt import BayesSearchCV
from sklearn.model_selection import StratifiedKFold

# SETTINGS - CHANGE THESE TO GET SOMETHING MEANINGFUL
ITERATIONS = 10 # 1000
#TRAINING_SIZE = 100000 # 20000000
#TEST_SIZE = 25000


# Classifier
bayes_cv_tuner = BayesSearchCV(
    estimator = xgb.XGBClassifier(
        n_jobs = 1,
        objective = 'multi:softmax',
        eval_metric = 'auc',
        silent=1,
        tree_method='approx'
    ),
    search_spaces = {
        'learning_rate': (0.01, 1.0, 'log-uniform'),
        'min_child_weight': (1, 20),
        'max_depth': (1, 30),
        'max_delta_step': (2, 20),
        'subsample': (0.01, 1.0, 'uniform'),
        'colsample_bytree': (0.01, 1.0, 'uniform'),
        'colsample_bylevel': (0.01, 1.0, 'uniform'),
        'reg_lambda': (1e-9, 1000, 'log-uniform'),
        'reg_alpha': (1e-9, 1.0, 'log-uniform'),
        'gamma': (1e-9, 0.5, 'log-uniform'),
        'min_child_weight': (0, 5),
        'n_estimators': (50, 5000),
        'scale_pos_weight': (1e-6, 500, 'log-uniform')
    },    
    #scoring = 'merror',
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=39
    ),
    n_jobs = 3,
    n_iter = ITERATIONS,   
    verbose = 0,
    refit = True,
    random_state = 42
)

def status_print(optim_result):
    """Status callback durring bayesian hyperparameter search"""
    
    # Get all the models tested so far in DataFrame format
    all_models = pd.DataFrame(bayes_cv_tuner.cv_results_)    
    
    # Get current parameters and the best parameters    
    best_params = pd.Series(bayes_cv_tuner.best_params_)
    print('Model #{}\nBest ROC-AUC: {}\nBest params: {}\n'.format(
        len(all_models),
        np.round(bayes_cv_tuner.best_score_, 4),
        bayes_cv_tuner.best_params_
    ))
    
    # Save all model results
    #clf_name = bayes_cv_tuner.estimator.__class__.__name__
    #all_models.to_csv(clf_name+"_cv_results.csv")


result = bayes_cv_tuner.fit(X4train.values, ym, callback=status_print)

bestp= bayes_cv_tuner.best_params_

clf2=xgb.XGBClassifier()
clf2.set_params(**bestp)

clf2.fit(X4train,ym)
pd.Series(clf2.feature_importances_,index=list(X4train.columns.values)).sort_values(ascending=True).head(50).plot(kind='barh',figsize=(12,18),title='XGBOOST FEATURE IMPORTANCE')

feati=gene+'_feat.png'
plt.savefig(feati)

mcpred=clf2.predict(X4train)
from sklearn.metrics import confusion_matrix
sns.heatmap(confusion_matrix(mcpred,ym),annot=True,fmt='2.0f')
confu=gene+'_conf.png'
plt.savefig(confu)


clf2.fit(X4train,ym)
pred=clf2.predict(X4test)
proba=clf2.predict_proba(X4test)

testset['Direction']=pred
testset['P_Direction']=proba[:,0]

clf2.fit(X4train,conf)
predc=clf2.predict(X4test)
probac=clf2.predict_proba(X4test)

testset['Confidence']=predc
testset['SE']=probac[:,0]

out=testset[['#chr', 'pos', 'ref','alt','gene', 'Direction','P_Direction', 'Confidence', 'SE']]
out.to_csv(outfile,sep="\t", index=False)

