import os
import numpy as np
import nibabel as nib
import pandas as pd
import joblib
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVR, LinearSVC
from sklearn.cross_validation import (train_test_split, cross_val_score,
    KFold)
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearSVC
from matplotlib import pylab as plt
import Seaborn as sns


OUT_DIR = os.path.abspath('social_deconf')
DECONF = True
DO_SAVEFIGS = True

try:
    os.mkdir(OUT_DIR)
except:
    print('Output directory already exists!')
    pass

# hierarchy of social brain networks as derived by Alcala-Lopez et al., 2017 Cereb Cortex
net1 = ['FG_L', 'FG_R', 'pSTS_L', 'pSTS_R', 'MTV5_L', 'MTV5_R']
net2 = ['AM_L', 'AM_R', 'HC_L', 'HC_R', 'vmPFC', 'NAC_L', 'NAC_R', 'rACC']
net3 = ['aMCC',  'AI_L',  'AI_R',  'SMG_L',  'SMG_R',  'SMA_L',  'SMA_R',  'IFG_L',  'IFG_R',  'Cereb_L',  'Cereb_R']
net4 = ['FP', 'dmPFC',  'PCC',  'TPJ_L',  'TPJ_R',  'Prec',  'MTG_L',  'MTG_R',  'TP_L',  'TP_R', 'pMCC']
net_hierarchy = OrderedDict(sensory=net1, limbic=net2, intermediate=net3, higherassociative=net4)

COLS_NAMES = []
for fname in ['ukbbids_social_brain.txt']:
    with open(fname) as f:
        lines=f.readlines()
        f.close()
        for line in lines:
            # if "(R)" in line:
            #     COLS_NAMES.append(line.split('\t'))
            COLS_NAMES.append(line.split('\t'))
COLS_NAMES = np.array(COLS_NAMES)

if 'ukbb' not in locals():
    ukbb = pd.read_csv('../UKBB/ukb_add1_holmes_merge_brain.csv')
else:
    print('Database is already in memory!')

T1_subnames, DMN_vols, rois = joblib.load('dumps/dump_sMRI_socialbrain_sym_r2.5_s5')
rois = np.array(rois)
T1_subnames_int = np.array([np.int(nr) for nr in T1_subnames], dtype=np.int64)
roi_names = np.array(rois)



head_size = StandardScaler().fit_transform(np.nan_to_num(ukbb['25006-2.0'].values[:, None]))
body_mass = StandardScaler().fit_transform(np.nan_to_num(ukbb['21001-0.0'].values[:, None]))
conf_mat = np.hstack([
    np.atleast_2d(head_size), np.atleast_2d(body_mass)])



# inds = np.searchsorted(T1_subnames_int, subids)
inds = np.searchsorted(T1_subnames_int, ukbb.eid)
inds_mri = []
source_array = T1_subnames_int
for _, sub in enumerate(ukbb.eid):
    i_found = np.where(sub == source_array)[0]
    if len(i_found) == 0:
        continue
    inds_mri.append(i_found[0])  # take first found subject
b_inds_ukbb = np.in1d(ukbb.eid, source_array[inds_mri])


print('%i matched matrices between data and UKBB found!' % np.sum(
        source_array[inds_mri] == ukbb.eid[b_inds_ukbb]))


# inds = np.searchsorted(T1_subnames_int, ukbb.eid)
T1_subnames = T1_subnames[inds_mri]
T1_subnames_int = T1_subnames_int[inds_mri]
DMN_vols = DMN_vols[inds_mri]
assert np.sum(T1_subnames_int == ukbb.eid[b_inds_ukbb].values) == len(inds_mri)

X = StandardScaler().fit_transform(DMN_vols)


TAR_COLS = COLS_NAMES[:, 0]


ukbb_tar = ukbb[TAR_COLS][b_inds_ukbb]


# replace numerical indicatros of missing response to NaNs
print('#NaNs before deleting non-responses: %i' %
        np.sum(np.isnan(ukbb_tar.values)))
ukbb_tar[ukbb_tar < 0] = np.nan
print('#NaNs after deleting non-responses: %i' %
        np.sum(np.isnan(ukbb_tar.values)))


# post-process categories with special encoding
ukbb_tar_ext = pd.get_dummies(ukbb_tar, columns=['6142-0.0'])

top10_jobs = ukbb_tar['22617-0.0'].value_counts().head(10).index  # top 10 jobs in UKBB brain
repl_dict = dict(zip(top10_jobs, np.arange(-110, -100)))
ukbb_tar_ext['22617-0.0'] = ukbb_tar_ext[
    '22617-0.0'].map(dict(zip(repl_dict, range(len(repl_dict)))))
ukbb_tar_ext = pd.get_dummies(ukbb_tar_ext, columns=['22617-0.0'])

top3_ethn = ukbb_tar['21000-0.0'].value_counts().head(3).index  # top 3 ethnicities in UKBB brain
repl_dict = dict(zip(top3_ethn, np.arange(-103, -100)))
ukbb_tar_ext['21000-0.0'] = ukbb_tar_ext[
    '21000-0.0'].map(dict(zip(repl_dict, range(len(repl_dict)))))

repl_dict = dict(zip(np.arange(1, 6), np.arange(-110, -105))) # encode freetime activities
ukbb_tar_ext['6160-0.0'] = ukbb_tar_ext[
    '6160-0.0'].map(dict(zip(repl_dict, range(len(repl_dict)))))
ukbb_tar_ext = pd.get_dummies(ukbb_tar_ext, columns=['6160-0.0'])

repl_dict = dict(zip(np.arange(1, 3), np.arange(-110, -108))) # encode household members
ukbb_tar_ext['6141-0.0'] = ukbb_tar_ext[
    '6141-0.0'].map(dict(zip(repl_dict, range(len(repl_dict)))))
ukbb_tar_ext = pd.get_dummies(ukbb_tar_ext, columns=['6141-0.0'])



ukbb_tar_ext = ukbb_tar_ext.dropna(thresh=53)  # at 10 missing items per individual
inds_nandrop = ukbb[b_inds_ukbb].index.isin(ukbb_tar_ext.index)  # indices of kept individuals


nan_cols = np.isnan(ukbb_tar_ext.values).sum(axis=0) > 0
nan_replaces = list(np.nanmedian(ukbb_tar_ext.values, axis=0))
for i_col, nan_col in enumerate(nan_cols):
    i_rows = np.where(np.isnan(ukbb_tar_ext.iloc[:, i_col]))[0]
    ukbb_tar_ext.iloc[i_rows, i_col] = nan_replaces[i_col]

assert np.isnan(ukbb_tar_ext).sum().sum() == 0



ukbb_tar_ss = StandardScaler().fit_transform(ukbb_tar_ext)
Y = ukbb_tar_ss.copy()
X = X[inds_nandrop]
T1_subnames = T1_subnames[inds_nandrop]
T1_subnames_int = T1_subnames_int[inds_nandrop]

assert len(Y) == len(X) == len(T1_subnames) == len(T1_subnames_int)

dict_id_2_descr = {id_:descr.split('\n')[0] for id_, descr in COLS_NAMES}



# sanity check prediction to make sure everything is well aligned
from sklearn.model_selection import cross_val_score
gender_mean_acc = cross_val_score(LinearSVC(), X, ukbb_tar_ext['31-0.0'].values, cv=3).mean()
print('Sanity check gender prediction is at %2.2f%%' % (gender_mean_acc * 100))



# Bayesian hierarchical regression: ROI volumes as lower-level input, gender on higher input -> output
import pymc3 as pm

ROBUST = False
REMOVE_OUTLIERS = True

cur_X = X.copy()
cur_conf = conf_mat.copy()[b_inds_ukbb][inds_nandrop]
names = roi_names


# output_name = 'aloneinhousehold+gender'
# output_name_short = output_name
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# cur_meta_cat = np.array(ukbb_tar_ext['709-0.0'].values, dtype=np.int)  # number of ppl in household
# cur_meta_cat = np.array(cur_meta_cat != 1, dtype=np.int)  # True=living more social since other ppl present
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# good_inds = np.where(cur_y > 0)[0]
# cur_X = cur_X[good_inds]
# cur_y = cur_y[good_inds]
# cur_conf = cur_conf[good_inds]
# cur_gender = cur_gender[good_inds]
# cur_meta_cat = cur_meta_cat[good_inds]
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# n_rare_class = np.min(np.bincount(cur_meta_cat))
# n_classes = len(np.unique(cur_meta_cat))
# print('DOWN-SAMPLING TO %i SUBJECTS WITH BALANCED CLASSES' % (n_rare_class * n_classes))
# balanced_cl_inds = []
# dw_rs = np.random.RandomState(0)
# for cl_label in np.unique(cur_meta_cat):
#     cl_inds = dw_rs.choice(np.where(cur_meta_cat == cl_label)[0], n_rare_class, replace=False)
#     balanced_cl_inds = balanced_cl_inds + [cl_inds]
# balanced_cl_inds = np.array(balanced_cl_inds).reshape(n_rare_class * n_classes)
# cur_X, cur_y, cur_meta_cat, cur_gender = (cur_X[balanced_cl_inds],
#     cur_y[balanced_cl_inds], cur_meta_cat[balanced_cl_inds], cur_gender[balanced_cl_inds])
# cur_conf = cur_conf[balanced_cl_inds]
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # separate subgroup for males
# subgroup_labels = ['female living alone in household', 'female living with other people in household',
#     'male living alone in household', 'male living with other people in household']

# output_name = 'manyinhousehold+gender'
# output_name_short = output_name
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# cur_meta_cat = np.array(ukbb_tar_ext['709-0.0'].values, dtype=np.int)  # number of ppl in household
# cur_meta_cat = np.array(cur_meta_cat > 2, dtype=np.int)  # True=living more social since other ppl present
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# good_inds = np.isfinite(cur_y)
# cur_X = cur_X[good_inds]
# cur_y = cur_y[good_inds]
# cur_conf = cur_conf[good_inds]
# cur_gender = cur_gender[good_inds]
# cur_meta_cat = cur_meta_cat[good_inds]
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# n_rare_class = np.min(np.bincount(cur_meta_cat))
# n_classes = len(np.unique(cur_meta_cat))
# print('DOWN-SAMPLING TO %i SUBJECTS WITH BALANCED CLASSES' % (n_rare_class * n_classes))
# balanced_cl_inds = []
# dw_rs = np.random.RandomState(0)
# for cl_label in np.unique(cur_meta_cat):
#     cl_inds = dw_rs.choice(np.where(cur_meta_cat == cl_label)[0], n_rare_class, replace=False)
#     balanced_cl_inds = balanced_cl_inds + [cl_inds]
# balanced_cl_inds = np.array(balanced_cl_inds).reshape(n_rare_class * n_classes)
# cur_X, cur_y, cur_meta_cat, cur_gender = (cur_X[balanced_cl_inds],
#     cur_y[balanced_cl_inds], cur_meta_cat[balanced_cl_inds], cur_gender[balanced_cl_inds])
# cur_conf = cur_conf[balanced_cl_inds]
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # separate subgroup for males
# subgroup_labels = ['female with 0-1 people in household', 'female with >=2 people in household',
#     'male with 0-1 people in household', 'male with >=2 people in household']

# output_name = 'hassiblings+gender'
# output_name_short = output_name
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_meta_cat = np.array(ukbb_tar_ext['5057-0.0'].values, dtype=np.int)  # number of siblings
# cur_meta_cat = np.array(cur_meta_cat != 0, dtype=np.int)  # True=living other ppl in same generation
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# good_inds = np.where(cur_y > 0)[0]
# cur_X = cur_X[good_inds]
# cur_y = cur_y[good_inds]
# cur_conf = cur_conf[good_inds]
# cur_gender = cur_gender[good_inds]
# cur_meta_cat = cur_meta_cat[good_inds]
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# n_rare_class = np.min(np.bincount(cur_meta_cat))
# n_classes = len(np.unique(cur_meta_cat))
# print('DOWN-SAMPLING TO %i SUBJECTS WITH BALANCED CLASSES' % (n_rare_class * n_classes))
# balanced_cl_inds = []
# dw_rs = np.random.RandomState(0)
# for cl_label in np.unique(cur_meta_cat):
#     cl_inds = dw_rs.choice(np.where(cur_meta_cat == cl_label)[0], n_rare_class, replace=False)
#     balanced_cl_inds = balanced_cl_inds + [cl_inds]
# balanced_cl_inds = np.array(balanced_cl_inds).reshape(n_rare_class * n_classes)
# cur_X, cur_y, cur_meta_cat, cur_gender = (cur_X[balanced_cl_inds],
#     cur_y[balanced_cl_inds], cur_meta_cat[balanced_cl_inds], cur_gender[balanced_cl_inds])
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # separate subgroup for males
# cur_conf = cur_conf[balanced_cl_inds]
# subgroup_labels = ['female without siblings', 'female has sibling',
#     'male without siblings', 'male has sibling']

# output_name = 'socialjob+gender'
# output_name_short = output_name
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_meta_cat = ukbb[b_inds_ukbb][inds_nandrop]['22617-0.0']  # job IDs
# top10jobs = cur_meta_cat.value_counts().head(10).index
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# good_inds1 = np.where(cur_y > 0)[0]
# good_inds2 = cur_meta_cat.isin(top10jobs).values
# good_inds = np.logical_and(good_inds1, good_inds2)
# cur_X = cur_X[good_inds]
# cur_y = cur_y[good_inds]
# cur_conf = cur_conf[good_inds]
# cur_gender = cur_gender[good_inds]
# cur_meta_cat = cur_meta_cat[good_inds]
# cur_meta_cat = np.where(
#     np.in1d(cur_meta_cat, [2314., 2315., 7111., 3211., 4123.]), 1, 0)
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # separate subgroup for males
# n_rare_class = np.min(np.bincount(cur_meta_cat))
# n_classes = len(np.unique(cur_meta_cat))
# # print('DOWN-SAMPLING TO %i SUBJECTS WITH BALANCED CLASSES' % (n_rare_class * n_classes))
# # balanced_cl_inds = []
# # dw_rs = np.random.RandomState(0)
# # for cl_label in np.unique(cur_meta_cat):
# #     cl_inds = dw_rs.choice(np.where(cur_meta_cat == cl_label)[0], n_rare_class, replace=False)
# #     balanced_cl_inds = balanced_cl_inds + [cl_inds]
# # balanced_cl_inds = np.array(balanced_cl_inds).reshape(n_rare_class * n_classes)
# # cur_X, cur_y, cur_meta_cat, cur_gender = (cur_X[balanced_cl_inds],
# # cur_y[balanced_cl_inds], cur_meta_cat[balanced_cl_inds], cur_gender[balanced_cl_inds])
# # cur_conf = cur_conf[balanced_cl_inds]
# # subgroup_labels = ['non-social job', 'social job']
# subgroup_labels = ['female without social job', 'female working with humans',
#     'male without social job', 'male working with humans']

# output_name = 'income+gender'
# output_name_short = output_name
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_meta_cat = np.array(ukbb_tar_ext['738-0.0'].values, dtype=np.int)  # income
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# good_inds1 = cur_y > 0
# good_inds2 = np.logical_or(cur_meta_cat == 1, cur_meta_cat == 5)  
# good_inds = np.logical_and(good_inds1, good_inds2)
# cur_X = cur_X[good_inds]
# cur_y = cur_y[good_inds]
# cur_gender = cur_gender[good_inds]
# cur_conf = cur_conf[good_inds]
# cur_meta_cat = cur_meta_cat[good_inds]
# cur_meta_cat = np.array(cur_meta_cat == 5, dtype=np.int)  # Less than 18,000 = 0; >100.000 = 1
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# n_rare_class = np.min(np.bincount(cur_meta_cat))
# n_classes = len(np.unique(cur_meta_cat))
# print('DOWN-SAMPLING TO %i SUBJECTS WITH BALANCED CLASSES' % (n_rare_class * n_classes))
# balanced_cl_inds = []
# dw_rs = np.random.RandomState(0)
# for cl_label in np.unique(cur_meta_cat):
#     cl_inds = dw_rs.choice(np.where(cur_meta_cat == cl_label)[0], n_rare_class, replace=False)
#     balanced_cl_inds = balanced_cl_inds + [cl_inds]
# balanced_cl_inds = np.array(balanced_cl_inds).reshape(n_rare_class * n_classes)
# cur_X, cur_y, cur_meta_cat, cur_gender = (cur_X[balanced_cl_inds],
#     cur_y[balanced_cl_inds], cur_meta_cat[balanced_cl_inds], cur_gender[balanced_cl_inds])
# cur_conf = cur_conf[balanced_cl_inds]
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # separate subgroup for males
# subgroup_labels = ['female low income', 'female with high income',
#     'male low income', 'male high income']

# output_name = 'privatehealthcare+gender'
# output_name_short = output_name
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_meta_cat = np.array(ukbb[b_inds_ukbb][inds_nandrop]['4674-2.0'].values, dtype=np.int)  # private healthcare
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# good_inds1 = cur_y > 0
# good_inds2 = cur_meta_cat > 0  # exclude non-responders
# good_inds = np.logical_and(good_inds1, good_inds2)
# cur_X = cur_X[good_inds]
# cur_y = cur_y[good_inds]
# cur_gender = cur_gender[good_inds]
# cur_conf = cur_conf[good_inds]
# cur_meta_cat = cur_meta_cat[good_inds]
# cur_meta_cat = np.array(cur_meta_cat <= 3, dtype=np.int)  # private ever = 1
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# n_rare_class = np.min(np.bincount(cur_meta_cat))
# n_classes = len(np.unique(cur_meta_cat))
# print('DOWN-SAMPLING TO %i SUBJECTS WITH BALANCED CLASSES' % (n_rare_class * n_classes))
# balanced_cl_inds = []
# dw_rs = np.random.RandomState(0)
# for cl_label in np.unique(cur_meta_cat):
#     cl_inds = dw_rs.choice(np.where(cur_meta_cat == cl_label)[0], n_rare_class, replace=False)
#     balanced_cl_inds = balanced_cl_inds + [cl_inds]
# balanced_cl_inds = np.array(balanced_cl_inds).reshape(n_rare_class * n_classes)
# cur_X, cur_y, cur_meta_cat, cur_gender = (cur_X[balanced_cl_inds],
#     cur_y[balanced_cl_inds], cur_meta_cat[balanced_cl_inds], cur_gender[balanced_cl_inds])
# cur_conf = cur_conf[balanced_cl_inds]
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # separate subgroup for males
# subgroup_labels = ['female with public health care', 'female with private health care',
#     'male with public health care', 'male with private health care']

# output_name = 'age_socialsupport+gender'
# output_name_short = output_name
# cur_meta_cat = np.array(ukbb_tar_ext['2110-0.0'].values == 5, dtype=np.int)  # soc. support
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # separate subgroup for males
# subgroup_labels = ['females w/ less social support', 'females w/ much social support',
#     'males w/ less social support', 'males w/ much social support']

# output_name = 'socialactivity+gender'
# output_name_short = output_name
# cur_meta_cat = np.array(ukbb[b_inds_ukbb][inds_nandrop]['6160-0.0'])
# cur_meta_cat[cur_meta_cat < 0] = np.nan
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# cur_meta_cat[cur_meta_cat < 0] = np.nan
# cur_meta_cat[cur_meta_cat > 4] = np.nan  # cut the "other" category
# good_inds = np.where(cur_meta_cat > 0)[0]
# cur_X = cur_X[good_inds]
# cur_y = cur_y[good_inds]
# cur_meta_cat = np.array(cur_meta_cat[good_inds] - 1, dtype=np.int)
# # cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# # cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # sEparate subgroup for males
# subgroup_labels = ['Sports club or gym', 'Pub or social club',
#     'Religious group', 'Adult education class']
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# cur_conf = np.hstack((cur_conf[good_inds], cur_gender[good_inds, np.newaxis]))

# break down the social activity into separate BHMs
# output_name = 'sportsgym+gender'
# output_name_short = output_name
# cur_meta_cat = np.array(ukbb[b_inds_ukbb][inds_nandrop]['6160-0.0'])
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# cur_meta_cat[cur_meta_cat < 0] = np.nan
# cur_meta_cat[cur_meta_cat > 4] = np.nan  # cut the "other" category
# good_inds = np.where(cur_meta_cat > 0)[0]
# cur_X = cur_X[good_inds]
# cur_y = cur_y[good_inds]
# cur_conf = cur_conf[good_inds]
# cur_gender = cur_gender[good_inds]
# cur_meta_cat = cur_meta_cat[good_inds]
# cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)  # Sports club or gym ?
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # sEparate subgroup for males
# n_classes = len(np.unique(cur_meta_cat))
# n_rare_class = np.min(np.bincount(cur_meta_cat))
# print('DOWN-SAMPLING TO %i SUBJECTS WITH BALANCED CLASSES' % (n_rare_class * n_classes))
# balanced_cl_inds = []
# dw_rs = np.random.RandomState(0)
# for cl_label in np.unique(cur_meta_cat):
#     cl_inds = dw_rs.choice(np.where(cur_meta_cat == cl_label)[0], n_rare_class, replace=False)
#     balanced_cl_inds = balanced_cl_inds + [cl_inds]
# balanced_cl_inds = np.array(balanced_cl_inds).reshape(n_rare_class * n_classes)
# assert balanced_cl_inds.shape[0] == n_rare_class * n_classes
# cur_X, cur_y, cur_meta_cat, cur_conf = (cur_X[balanced_cl_inds],
#     cur_y[balanced_cl_inds], cur_meta_cat[balanced_cl_inds], cur_conf[balanced_cl_inds])
# subgroup_labels = ['female not in sports club', 'female in sports club',
#     'male not in sports club', 'male in sports club']

# output_name = 'anysocialact+gender'
# output_name_short = output_name
# cur_meta_cat = np.array(ukbb[b_inds_ukbb][inds_nandrop]['6160-0.0'])
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# cur_meta_cat[cur_meta_cat == -7] = 7  # to make our target positive
# cur_meta_cat[cur_meta_cat < 0] = np.nan
# good_inds = np.where(cur_meta_cat > 0)[0]
# cur_X = cur_X[good_inds]
# cur_y = cur_y[good_inds]
# cur_conf = cur_conf[good_inds]
# cur_gender = cur_gender[good_inds]
# cur_meta_cat = cur_meta_cat[good_inds]
# cur_meta_cat = np.array(cur_meta_cat != 7, dtype=np.int)  # 0=no socialact; 1=any social act
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # sEparate subgroup for males
# n_classes = len(np.unique(cur_meta_cat))
# n_rare_class = np.min(np.bincount(cur_meta_cat))
# print('DOWN-SAMPLING TO %i SUBJECTS WITH BALANCED CLASSES' % (n_rare_class * n_classes))
# balanced_cl_inds = []
# dw_rs = np.random.RandomState(0)
# for cl_label in np.unique(cur_meta_cat):
#     cl_inds = dw_rs.choice(np.where(cur_meta_cat == cl_label)[0], n_rare_class, replace=False)
#     balanced_cl_inds = balanced_cl_inds + [cl_inds]
# balanced_cl_inds = np.array(balanced_cl_inds).reshape(n_rare_class * n_classes)
# assert balanced_cl_inds.shape[0] == n_rare_class * n_classes
# cur_X, cur_y, cur_meta_cat, cur_conf = (cur_X[balanced_cl_inds],
#     cur_y[balanced_cl_inds], cur_meta_cat[balanced_cl_inds], cur_conf[balanced_cl_inds])
# subgroup_labels = ['female without weekly social activity', 'female with some social activity',
#     'male without weekly social activity', 'male with some social activity']

# output_name = 'sexpartners+gender'
# output_name_short = output_name
# cur_meta_cat = np.array(ukbb[b_inds_ukbb][inds_nandrop]['2149-0.0'], dtype=np.int)
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# good_inds = np.logical_and(np.isfinite(cur_y), cur_meta_cat > 0)  # cut the non-response categories
# cur_X = cur_X[good_inds]
# cur_y = cur_y[good_inds]
# cur_conf = cur_conf[good_inds]
# cur_gender = cur_gender[good_inds]
# cur_meta_cat = cur_meta_cat[good_inds]
# cur_meta_cat = np.array(cur_meta_cat == 1, dtype=np.int)  # true=1 lifetime sexual partner
# n_classes = len(np.unique(cur_meta_cat))
# n_rare_class = np.min(np.bincount(cur_meta_cat))
# print('DOWN-SAMPLING TO %i SUBJECTS WITH BALANCED CLASSES' % (n_rare_class * n_classes))
# balanced_cl_inds = []
# dw_rs = np.random.RandomState(0)
# for cl_label in np.unique(cur_meta_cat):
#     cl_inds = dw_rs.choice(np.where(cur_meta_cat == cl_label)[0], n_rare_class, replace=False)
#     balanced_cl_inds = balanced_cl_inds + [cl_inds]
# balanced_cl_inds = np.array(balanced_cl_inds).reshape(n_rare_class * n_classes)
# assert balanced_cl_inds.shape[0] == n_rare_class * n_classes
# cur_X, cur_y, cur_meta_cat = (cur_X[balanced_cl_inds],
#     cur_y[balanced_cl_inds], cur_meta_cat[balanced_cl_inds])
# cur_conf = cur_conf[balanced_cl_inds]
# cur_gender = cur_gender[balanced_cl_inds]
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # sEparate subgroup for males
# subgroup_labels = ['female with >1 partner', 'female with 1 lifetime partner',
#     'male with >1 partner', 'male with 1 lifetime partner']

# output_name = 'householdrelation+gender'
# output_name_short = output_name
# cur_meta_cat = ukbb[b_inds_ukbb][inds_nandrop]['6141-0.0'].values  # relation to other ppl in household
# good_inds = np.logical_or(cur_meta_cat==1, cur_meta_cat==2)
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# cur_X = cur_X[good_inds]
# cur_y = cur_y[good_inds]
# cur_gender = cur_gender[good_inds]
# cur_conf = cur_conf[good_inds]
# cur_meta_cat = np.array(cur_meta_cat[good_inds] - 1, np.int)
# n_rare_class = np.min(np.bincount(cur_meta_cat))
# print('DOWN-SAMPLING TO %i SUBJECTS WITH BALANCED CLASSES' % (n_rare_class * 2))
# dw_rs = np.random.RandomState(0)
# zero_cl_inds = dw_rs.choice(np.where(cur_meta_cat == 0)[0], n_rare_class, replace=False)
# one_cl_inds = dw_rs.choice(np.where(cur_meta_cat == 1)[0], n_rare_class, replace=False)
# balanced_cl_inds = np.hstack((zero_cl_inds, one_cl_inds))
# cur_X, cur_y, cur_meta_cat, cur_gender = (cur_X[balanced_cl_inds],
#     cur_y[balanced_cl_inds], cur_meta_cat[balanced_cl_inds], cur_gender[balanced_cl_inds])
# cur_conf = cur_conf[balanced_cl_inds]
# subgroup_labels = ['lives with partner', 'lives with children']
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # separate subgroup for males
# subgroup_labels = ['female lives w/ partner', 'female lives w/ kids',
#     'male lives w/ partner', 'male lives w/ kids']

# output_name = 'familysatisfaction+gender'
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# cur_meta_cat = ukbb_tar_ext['4559-0.0'].values.astype(np.int)  # family relationship satisfaction
# # inds_answer2 = np.where(cur_meta_cat == 2)[0]
# # rs = np.random.RandomState(1)
# # inds_answer2 = rs.choice(inds_answer2, 962, replace=False)
# # inds_notanswer2 = np.where(cur_meta_cat != 2)[0]
# # inds_subsel = np.hstack((inds_answer2, inds_notanswer2))
# inds_subsel = np.where(cur_meta_cat > 0)[0]
# cur_X = cur_X[inds_subsel]
# cur_y = cur_y[inds_subsel]
# cur_conf = cur_conf[inds_subsel]
# cur_meta_cat = cur_meta_cat[inds_subsel]
# cur_meta_cat = np.array(cur_meta_cat <= 2, np.int)  # 1=happy
# n_rare_class = np.min(np.bincount(cur_meta_cat))
# print('DOWN-SAMPLING TO %i SUBJECTS WITH BALANCED CLASSES' % (n_rare_class * 2))
# dw_rs = np.random.RandomState(0)
# zero_cl_inds = dw_rs.choice(np.where(cur_meta_cat == 0)[0], n_rare_class, replace=False)
# one_cl_inds = dw_rs.choice(np.where(cur_meta_cat == 1)[0], n_rare_class, replace=False)
# balanced_cl_inds = np.hstack((zero_cl_inds, one_cl_inds))
# cur_X, cur_y, cur_meta_cat = (cur_X[balanced_cl_inds],
#     cur_y[balanced_cl_inds], cur_meta_cat[balanced_cl_inds])
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# cur_gender = cur_gender[inds_subsel][balanced_cl_inds]
# cur_conf = cur_conf[balanced_cl_inds]
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # separate subgroup for males
# subgroup_labels = ['females not happy with family', 'females happy with family',
#     'males not happy with family', 'males happy with family']

# output_name = 'friendshipsatisfaction+gender'
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# cur_meta_cat = ukbb_tar_ext['4570-0.0'].values.astype(np.int)  # friendships satisfaction
# cur_meta_cat = np.array(cur_meta_cat <= 2, np.int)  # 1=happy
# n_rare_class = np.min(np.bincount(cur_meta_cat))
# print('DOWN-SAMPLING TO %i SUBJECTS WITH BALANCED CLASSES' % (n_rare_class * 2))
# dw_rs = np.random.RandomState(0)
# zero_cl_inds = dw_rs.choice(np.where(cur_meta_cat == 0)[0], n_rare_class, replace=False)
# one_cl_inds = dw_rs.choice(np.where(cur_meta_cat == 1)[0], n_rare_class, replace=False)
# balanced_cl_inds = np.hstack((zero_cl_inds, one_cl_inds))
# cur_X, cur_y, cur_meta_cat = (cur_X[balanced_cl_inds],
#     cur_y[balanced_cl_inds], cur_meta_cat[balanced_cl_inds])
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# cur_gender = cur_gender[balanced_cl_inds]
# cur_conf = cur_conf[balanced_cl_inds]
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # separate subgroup for males
# subgroup_labels = ['females not happy with friends', 'females happy with friends',
#     'males not happy with friends', 'males happy with friends']

# output_name = 'loneliness+gender'
# cur_y = ukbb[b_inds_ukbb][inds_nandrop]['21022-0.0'].values  # age at recruitment
# cur_y = np.squeeze(StandardScaler().fit_transform(cur_y[:, None]))
# cur_meta_cat = ukbb[b_inds_ukbb][inds_nandrop]['2020-0.0'].values  # loneliness
# cur_meta_cat[cur_meta_cat < 0] = np.nan  # take out lacking response status
# cur_gender = ukbb[b_inds_ukbb][inds_nandrop]['31-0.0'].values.astype(np.int)  # gender
# inds_nandrop2 = np.isfinite(cur_meta_cat)
# cur_y = cur_y[inds_nandrop2]
# cur_X = cur_X[inds_nandrop2]
# cur_meta_cat = cur_meta_cat[inds_nandrop2]
# cur_gender = cur_gender[inds_nandrop2]
# cur_conf = cur_conf[inds_nandrop2]
# cur_meta_cat = np.array(cur_meta_cat, dtype=np.int)
# n_rare_class = np.min(np.bincount(cur_meta_cat))
# print('DOWN-SAMPLING TO %i SUBJECTS WITH BALANCED CLASSES' % (n_rare_class * 2))
# dw_rs = np.random.RandomState(0)
# zero_cl_inds = dw_rs.choice(np.where(cur_meta_cat == 0)[0], n_rare_class, replace=False)
# one_cl_inds = dw_rs.choice(np.where(cur_meta_cat == 1)[0], n_rare_class, replace=False)
# balanced_cl_inds = np.hstack((zero_cl_inds, one_cl_inds))
# cur_X, cur_y, cur_meta_cat = (cur_X[balanced_cl_inds],
#     cur_y[balanced_cl_inds], cur_meta_cat[balanced_cl_inds])
# cur_conf = cur_conf[balanced_cl_inds]
# cur_gender = cur_gender[balanced_cl_inds]
# cur_meta_cat[np.logical_and(cur_meta_cat==0, cur_gender==1)] += 2  # separate subgroup for males
# cur_meta_cat[np.logical_and(cur_meta_cat==1, cur_gender==1)] += 2  # separate subgroup for males
# subgroup_labels = ['females not feeling lonely', 'loneley females',
#     'males not feeling lonely', 'lonely males']


if REMOVE_OUTLIERS:
    OUTLIER_Z = 2.5
    inds_nooutlier = np.mean(
            np.logical_or(cur_X > OUTLIER_Z, cur_X < -OUTLIER_Z), axis=1) == 0
    n_outliers = len(inds_nooutlier) - np.sum(inds_nooutlier)
    print('Removing %i outlier observations X!!!' % n_outliers)

    cur_X, cur_y, cur_meta_cat = cur_X[inds_nooutlier], cur_y[inds_nooutlier], cur_meta_cat[inds_nooutlier]
    cur_conf = cur_conf[inds_nooutlier]
    # cur_gender = cur_gender[inds_nooutlier]
    output_name += 'O' 


n_meta_cat = len(np.unique(cur_meta_cat))


print('%s: The high-level subgroup distribution is:' % output_name)
print(np.bincount(cur_meta_cat))





pm_varnames = []
with pm.Model() as hierarchical_model:

    roi_betas, cur_net_mus, cur_net_sigma_bs = [], [], []
    for i_net_mid_HP, net_key in enumerate(net_hierarchy.keys()):
        cur_net_mu = pm.Normal('mu_%s' % net_key, mu=0., sd=1, shape=n_meta_cat)
        cur_net_sigma_b = pm.HalfCauchy('sigma_%s' % net_key, 1, shape=n_meta_cat)
        pm_varnames.append('mu_%s' % net_key)
        pm_varnames.append('sigma_%s' % net_key)
        cur_net_mus.append(cur_net_mu)
        cur_net_sigma_bs.append(cur_net_sigma_b)

        net_rois = net_hierarchy[net_key]
        for i_net_roi, net_roi in enumerate(net_rois):
            behav_name = net_roi
            pm_varnames.append(behav_name)
            cur_beta_param = pm.Normal(behav_name, mu=cur_net_mus[-1], sd=cur_net_sigma_bs[-1],
                shape=n_meta_cat)
            i_roi_in_X = np.where(roi_names == net_roi)[0][0]  # double zero crucial!!! ...wierd crashes otherwise
            print('Adding %s ROI for network %s (X index %i)' % (behav_name, net_key, i_roi_in_X))
            print(cur_X[:, i_roi_in_X].shape)
            roi_betas.append(cur_beta_param)
            if i_net_mid_HP == 0 and i_net_roi == 0:
                beh_est = roi_betas[-1][cur_meta_cat] * cur_X[:, i_roi_in_X]
            else:
                beh_est = beh_est + roi_betas[-1][cur_meta_cat] * cur_X[:, i_roi_in_X]

    for i_deconf_var, cur_bad_var in enumerate(cur_conf.T):
        var_dist = pm.Normal('nuisance_var%i' % i_deconf_var, mu=0., sd=1, shape=1)
        best_est = beh_est + var_dist * cur_bad_var
        pm_varnames.append('nuisance_var%i' % i_deconf_var)
    print('Confounding variables: the GLM contains %i variables of no interest' % len(cur_conf.T))


    # define data likelihood
    if ROBUST:
        ## define prior for Student T degrees of freedom
        nu_out = pm.Uniform('nu_out', lower=1, upper=100)
        tau_out = pm.Gamma('tau_out', .01, .01)
        group_like = pm.StudentT('beh_like', mu=beh_est, lam=tau_out, nu=nu_out, observed=cur_y)
        output_name += 'R'
    else:
        eps = pm.HalfCauchy('eps', 5)  # Model error
        group_like = pm.Normal('beh_like', mu=beh_est, sd=eps, observed=cur_y)


with hierarchical_model:
    hierarchical_trace = pm.sample(draws=5000, n_init=1000, #init='advi',
        chains=1, cores=1, progressbar=True,
        random_seed=[123]  # one per chain needed
        )

for cur_roi in pm_varnames:
    from matplotlib.lines import Line2D
    plt.close('all')
    THRESH = 0.5
    n_last_chains = 1000
    try:
        fig = pm.plot_posterior(hierarchical_trace[-n_last_chains:], varnames=[cur_roi])
        try:
            for i_higher_cat in range(n_meta_cat):
                fig[i_higher_cat].set_xlim(-THRESH, THRESH)  # make plots more comparable
        except:
            pass
        plt.tight_layout()
        plt.savefig('%s/%s_%s_posterior.png' % (OUT_DIR, output_name, cur_roi), dpi=150)

        fig = pm.traceplot(hierarchical_trace[-n_last_chains:], varnames=[cur_roi])
        max_abs_mode = np.max(np.abs(hierarchical_trace[-n_last_chains:][cur_roi].mean(0)))
        try:
            if max_abs_mode < THRESH and not 'nuisance' in cur_roi:
                fig[0][0].set_xlim(-THRESH, THRESH)  # make plots more comparable
        except:
            pass
        post_lines = fig[0][0].get_lines()
        custom_lines = [Line2D([0], [0], color=l.get_c(), lw=4) for l in post_lines]
        if subgroup_labels is None:
            subgroup_labels = ['subgroup %i' % i for i in range(len(custom_lines))]
        fig[0][0].legend(custom_lines, subgroup_labels, loc='upper left', prop={'size': 7.5})
        plt.savefig('%s/%s_%s.png' % (OUT_DIR, output_name, cur_roi), dpi=150)
    except:
        pass

t = hierarchical_trace

from sklearn.metrics import r2_score
Y_ppc_insample = pm.sample_ppc(hierarchical_trace, 5000, hierarchical_model, random_seed=123)['beh_like']
y_pred_insample = Y_ppc_insample.mean(axis=0)
ppc_insample = r2_score(cur_y, y_pred_insample)
out_str = 'PPC in sample R^2: %2.6f' % (ppc_insample)
print(out_str)
plt.figure(figsize=(7, 8))
sns.regplot(x=cur_y, y=y_pred_insample, fit_reg=True, ci=95,
    line_kws={'color':'black', 'linewidth':4})
plt.xlabel('real output variable')
plt.ylabel('predicted output variable')
plt.title(out_str + ' (%i samples)' % len(cur_X))
plt.savefig('%s/%s_r2scatter.pdf' % (OUT_DIR, output_name), dpi=150)

plt.figure()
plt.hist([it.mean() for it in Y_ppc_insample.T], bins=19, alpha=0.35,
    label='predicted output')
plt.hist(cur_y, bins=19, alpha=0.5, label='original output')
plt.legend(loc='upper right')
plt.title('Posterior predictive check: predictive distribution', fontsize=10)
plt.savefig('%s/%s_ppc.pdf' % (OUT_DIR, output_name), dpi=150)


joblib.dump([hierarchical_trace, hierarchical_model], os.path.join(OUT_DIR, output_name + '_dump'), compress=9)



