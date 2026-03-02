#%%
import pandas as pd
import numpy as np
from optbinning import BinningProcess
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
# from fgclustering import FgClustering
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import _tree
#%%
def encode_row(value_str, unique_values):
    one_hot = [0] * len(unique_values)
    if pd.isna(value_str) or value_str in ('nan', ''):
        return [np.nan] * len(unique_values)
    for part in str(value_str).split('|'):
        try:
            idx = unique_values.index(int(float(part)))
            one_hot[idx] = 1
        except (ValueError, IndexError):
            continue
    return one_hot

#%%
def encode_var(data_df, var):
    df_ = data_df[var].astype(str)
    unique_values = set()
    for val in df_:
        if pd.isna(val) or val in ('nan', ''):
            continue
        for part in str(val).split('|'):
            try:
                unique_values.add(int(float(part)))
            except ValueError:
                pass

    # Sort and create list for column order
    unique_values = sorted(unique_values)
    print(f"({var}) Unique values:", unique_values)
    # Apply encoding to each row, expanding into separate columns
    one_hot_cols = df_.apply(lambda x: encode_row(x, unique_values))
    one_hot_df = pd.DataFrame(one_hot_cols.tolist(), columns=[f'{var}_{v}' for v in unique_values])
    # print(one_hot_df)
    return one_hot_df

# %%
# Load data from ADNI dataset
data_df = pd.read_csv('data/All_Subjects_My_Table_03Jul2025.csv')
mri_df = pd.read_csv('data/UCSFFSX7_20Jun2025.csv')
# vars_df = pd.read_csv('adni_variables.csv')

#%%
DXMOTHET_df = encode_var(data_df, 'DXMOTHET')
KEYMED_df = encode_var(data_df, 'KEYMED')
PTNLANG_df = encode_var(data_df, 'PTNLANG')
MHNUM_df = encode_var(data_df, 'MHNUM')
PTMARRY_df = encode_var(data_df, 'PTMARRY')
PTHOME_df = encode_var(data_df, 'PTHOME')
PTNOTRT_df = encode_var(data_df, 'PTNOTRT')
PTPLANG_df = encode_var(data_df, 'PTPLANG')
data_df = pd.concat([data_df,DXMOTHET_df, KEYMED_df, PTNLANG_df, MHNUM_df, PTMARRY_df, PTHOME_df, PTNOTRT_df, PTPLANG_df], axis=1)

#%%
data_df.loc[:,'RCT20'] = pd.to_numeric(data_df.loc[:,'RCT20'], errors='coerce')
data_df.loc[:,'RCT1408'] = pd.to_numeric(data_df.loc[:,'RCT1408'], errors='coerce')
data_df.loc[:,'CV'] = pd.DataFrame(data_df.loc[:,'CV'].astype(str).str.split('%').to_list())[0].astype(float)
data_df.loc[:,'RCT19'] = pd.to_numeric(data_df.loc[:,'RCT19'], errors='coerce')
data_df.loc[:,'HMT40'] = pd.to_numeric(data_df.loc[:,'HMT40'], errors='coerce')
data_df.loc[:,'BAT126'] = pd.to_numeric(data_df.loc[:,'BAT126'], errors='coerce')
data_df.loc[:,'RCT392'] = pd.to_numeric(data_df.loc[:,'RCT392'], errors='coerce')
data_df.loc[:,'RCT14'] = pd.to_numeric(data_df.loc[:,'RCT14'], errors='coerce')

#%%
# Calculate years from retirement
# data_df['diff_PTRTYR']=pd.to_datetime(data_df['subject_date']).dt.year-data_df['PTRTYR']

#%%
# Calculate BMI
data_df['HEIGHT'] = data_df['VSHEIGHT'].where(data_df['VSHTUNIT']==2,data_df['VSHEIGHT']*2.54)
data_df['WEIGHT'] = data_df['VSWEIGHT'].where(data_df['VSWTUNIT']==2,data_df['VSWEIGHT']*0.453592)
data_df['BMI'] = data_df['WEIGHT']/((data_df['HEIGHT']/100)**2)

#%%
datatypes_dict = {
'ABETA42':float,'BAT126':float,'CV':float,'DILUTION_CORRECTED_CONC':float,'PTAU':float,'RCT14':float,'RCT392':float,
'TAU':float,'BCPREDX':'Int64','CDCARE':'Int64','CDHOME':float,'CDRSB':float,'DXDSEV':'Int64','DXMDUE':'Int64','DXMOTHET':str,
'DXMOTHET_1':'Int64','DXMOTHET_2':'Int64','DXMOTHET_4':'Int64','DXMOTHET_5':'Int64','DXMOTHET_6':'Int64','DXMOTHET_7':'Int64', 
'DXMOTHET_9':'Int64','DXMOTHET_11':'Int64','DXMOTHET_12':'Int64','DXMOTHET_14':'Int64','DXMPTR1':'Int64','DXMPTR2':'Int64',
'DXMPTR3':'Int64','DXMPTR4':'Int64','DXMPTR5':'Int64','FAQFINAN':'Int64','FAQGAME':'Int64','FAQSHOP':'Int64','FAQTOTAL':'Int64',
'FAQTRAVL':'Int64','KEYMED':str,'KEYMED_0':'Int64','KEYMED_1':'Int64','KEYMED_2':'Int64','KEYMED_3':'Int64','KEYMED_4':'Int64',
'KEYMED_5':'Int64','KEYMED_6':'Int64','KEYMED_7':'Int64','LDELTOTAL':'Int64','LIMMTOTAL':'Int64','MMSCORE':'Int64','NXGAIT':'Int64',
'PTCOGBEG':'Int64','TOTAL13':float,'TOTSCORE':float,'TRAASCOR':'Int64','PTEDUCAT':'Int64','PTENGSPK':'Int64','PTHOME':'Int64',
'PTLANGTTL':'Int64','PTMARRY':'Int64','PTNLANG':str,'PTNLANG_1':'Int64', 'PTNLANG_2':'Int64', 'PTNLANG_3':'Int64','PTNOTRT':'Int64',
'PTPLANG':'Int64','PTRTYR':'Int64','PTSPOTTIM':float,'PTWORK':'Int64','DIAGNOSIS':'Int64','MRI_acquired':'Int64','research_group':str,
'GENOTYPE':str,'PTGENDER':'Int64','AXDPMOOD':'Int64','BCDEPRES':'Int64','BCDPMOOD':'Int64','DXDEP':'Int64','DXNODEP':'Int64',
'GDTOTAL':'Int64','GLUCOSE':float,'HMHYPERT':'Int64','HMT40':float,'NPID':'Int64','NPID8':'Int64','NPIDSEV':'Int64','NPIDTOT':'Int64',
'NPIK':'Int64','NPIK1':'Int64','NPIK2':'Int64','NPIK3':'Int64','NPIK4':'Int64','NPIK5':'Int64','NPIK6':'Int64','NPIK7':'Int64',
'NPIK8':'Int64','NPIK9A':'Int64','NPIK9B':'Int64','NPIK9C':'Int64','NPIKSEV':'Int64','NPIKTOT':'Int64','RCT1408':float,'RCT19':float,
'RCT20':float,'VSBPDIA':float,'VSBPSYS':float,'VSHEIGHT':float,'VSHTUNIT':'Int64','VSWEIGHT':float,'VSWTUNIT':'Int64','EXABUSE':'Int64',
'MH14AALCH':'Int64','MH14ALCH':'Int64','MH16ASMOK':float,'MH16CSMOK':float,'MH16SMOK':'Int64','AXCHEST':'Int64','AXFALL':'Int64',
'AXHDACHE':'Int64','AXINSOMN':'Int64','AXMUSCLE':'Int64','AXVISION':'Int64','BCCHEST':'Int64','BCFALL':'Int64','BCHDACHE':'Int64',
'BCINSOMN':'Int64','BCMUSCLE':'Int64','BCSTROKE':'Int64','BCVISION':'Int64','BSXCHRON':'Int64','BSXSEVER':'Int64','BSXSYMNO':'Int64',
'HMSCORE':'Int64','HMSOMATC':'Int64','HMSTROKE':'Int64','INCVISUAL':'Int64','MH12RENA':'Int64','MH4CARD':'Int64','MHCUR':'Int64',
'MHNUM':'Int64','MHSTAB':'Int64','NPPDXI':'Int64','NPPDXJ':'Int64','NXAUDITO':'Int64','PXHEART':'Int64','PXPERIPH':'Int64',
'MHNUM_1':'Int64','MHNUM_2':'Int64','MHNUM_3':'Int64','MHNUM_4':'Int64','MHNUM_5':'Int64','MHNUM_6':'Int64','MHNUM_7':'Int64',
'MHNUM_8':'Int64','MHNUM_9':'Int64','MHNUM_10':'Int64','MHNUM_11':'Int64','MHNUM_12':'Int64','MHNUM_13':'Int64','MHNUM_14':'Int64',
'MHNUM_15':'Int64','MHNUM_16':'Int64','MHNUM_17':'Int64','MHNUM_18':'Int64','MHNUM_19':'Int64',
}

# %%
# Filter subjects based on the research group
# Research group who was initially identified as Controls
data_cn_df = data_df[data_df['research_group']=='CN']
subjects_cn_id = data_cn_df['subject_id'].unique()

# Research group who was initially identified as MCI
# data_mci_df = data_df[data_df['research_group']=='MCI']
# subjects_mci_id = data_mci_df['subject_id'].unique()
# %%
# For each subject in the Control research group, check if he/she has multiple diagnosis
# If the subject had multiple diagnosis (mci or ad) include him/her in a list
subjects_multiple_diagnosis_cn = []
for subject in subjects_cn_id:
    n_diagnosis = len(data_df[data_df['subject_id']==subject]['DIAGNOSIS'].dropna().unique())
    if n_diagnosis>1:
        subjects_multiple_diagnosis_cn.append(subject)
# Create a dataframe with all the subjects who had multiple diagnosis 
data_cn_change_df = data_df[data_df['subject_id'].isin(subjects_multiple_diagnosis_cn)]
# %%
# For each subject in the Control research group, check if he/she has only one diagnosis
# If the subject had only one diagnosis, include him/her in a list
subjects_one_diagnosis_cn = []
for subject in subjects_cn_id:
    n_diagnosis = len(data_df[data_df['subject_id']==subject]['DIAGNOSIS'].dropna().unique())
    if n_diagnosis==1:
        subjects_one_diagnosis_cn.append(subject)
# Create a dataframe with all the subjects who had only one diagnosis 
data_cn_no_change_df = data_df[data_df['subject_id'].isin(subjects_one_diagnosis_cn)]

# %%
timepoints = ['sc','bl']
timepoints_all = ['sc','bl','4_sc','4_bl']
filtered_data = []
filtered_mri_data = {}
for subject_id in subjects_multiple_diagnosis_cn:
    # Pick a subject from the group that transition from normal to mci or ad
    subject_df = data_cn_change_df[data_cn_change_df['subject_id']==subject_id]
    subject_mri_df = mri_df[mri_df['PTID']==subject_id]
    # Take screening and baseline timepoints information
    cond_baseline = (subject_df['visit'].isin(timepoints))
    cond_mri_baseline = (subject_mri_df['VISCODE2'].isin(timepoints))
    # If information at baseline timepoint is missing, fill it with information from screening timepoint
    baseline_df = subject_df[cond_baseline].sort_values(by=['visit'],ascending=False).infer_objects(copy=False).bfill(axis=0).ffill(axis=0).iloc[-1:,:]
    baseline_mri_df = subject_mri_df[cond_mri_baseline].sort_values(by=['VISCODE2'],ascending=False).iloc[-1:,:]
    # Check if we have more information at other timepoionts other than baseline or screening
    filtered_mri_data.update({subject_id:[baseline_mri_df['VISCODE2'],baseline_df['visit']]})
    cond_all = (subject_df['visit'].isin(timepoints_all))
    if len(subject_df[~cond_all])>1:
        # Obtain the timepoints of other visits (in months) 
        months = subject_df[~cond_baseline]['visit'].str.split('m',expand=True)[1].dropna().astype(int).sort_values()
        subject_df_months = subject_df.loc[months.index]
        subject_df_months['months'] = months
        subject_df_months = subject_df_months.dropna(subset=['DIAGNOSIS'])
        if len(subject_df_months)>0: 
            # Check if the change in diagnosis is less than or equal to 12 months after baseline measurements
            diagnosis_cn_before_12_months = (subject_df_months[subject_df_months['months']<=12]['DIAGNOSIS']==1).all()
            # Take last diagnosis 
            diagnosis_last_visit = subject_df_months.iloc[-1]['DIAGNOSIS']
            # If the change in diagnosis occured after 12 months and last diagnosis is different from normal,
            # keep the records
            if diagnosis_cn_before_12_months and (diagnosis_last_visit!=1):
                baseline_df['study_duration'] = months.max()
                baseline_df['last_diagnosis'] = diagnosis_last_visit
                filtered_data.append(baseline_df)
subjects_transition_df = pd.concat(filtered_data).astype(datatypes_dict)
subjects_transition_df['transition'] = 1    

# %%
filtered_data = []
for subject_id in subjects_one_diagnosis_cn:
    # Pick a subject from the group that do not transition
    subject_df = data_cn_no_change_df[data_cn_no_change_df['subject_id']==subject_id]
    # Take screening and baseline timepoints information
    cond_baseline = (subject_df['visit'].isin(timepoints))
    # If information at baseline timepoint is missing, fill it with information from screening timepoint
    baseline_df = subject_df[cond_baseline].sort_values(by=['visit'],ascending=False).infer_objects(copy=False).bfill(axis=0).ffill(axis=0).iloc[-1:,:]
    # Check if we have more information at other timepoints other than baseline or screening
    cond_all = (subject_df['visit'].isin(timepoints_all))
    if len(subject_df[~cond_all])>1:
        # Obtain the timepoints of other visits (in months) 
        months = subject_df[~cond_baseline]['visit'].str.split('m',expand=True)[1].dropna().astype(int).sort_values()
        subject_df_months = subject_df.loc[months.index]
        subject_df_months['months'] = months
        subject_df_months = subject_df_months.dropna(subset=['DIAGNOSIS'])
        if len(subject_df_months)>0:
            # Take last diagnosis 
            diagnosis_last_visit = subject_df_months.iloc[-1]['DIAGNOSIS']
            baseline_df['study_duration'] = months.max()
            baseline_df['last_diagnosis'] = diagnosis_last_visit
            filtered_data.append(baseline_df)
subjects_no_transition_df = pd.concat(filtered_data).astype(datatypes_dict)
subjects_no_transition_df['transition'] = 0

# %%
# Look for subjects in the 'no transition group' who have similar attributes to the ones in the 'transition group'
selected_pairs = []
num_vars_names = ['subject_age']
cat_vars_names = ['PTGENDER','GENOTYPE']
subjects_id_transition = subjects_transition_df['subject_id']
for subject in subjects_id_transition:
    # Check age, sex, and duration of study, and rank candidates based on the norm of the difference
    num_vars_transition = subjects_transition_df[subjects_transition_df['subject_id']==subject][num_vars_names]
    cat_vars_transition = subjects_transition_df[subjects_transition_df['subject_id']==subject][cat_vars_names]
    diff_df = (subjects_no_transition_df[num_vars_names]-num_vars_transition.values).astype(float)
    diff_df['norm'] = np.linalg.norm(diff_df,axis=1)
    diff_df = diff_df.sort_values(by=['norm'], ascending=True)
    diff_df = pd.merge(left=diff_df, right=subjects_no_transition_df[cat_vars_names], left_index=True, right_index=True)
    cat_vars_sim_idx = np.all(diff_df[cat_vars_names]==cat_vars_transition.values, axis=1)
    canditates_df = diff_df.loc[cat_vars_sim_idx]
    best_candidate = subjects_no_transition_df.loc[[canditates_df.index[0]]]
    selected_pairs.append(best_candidate)
    # Remove subject from the pool
    subjects_no_transition_df = subjects_no_transition_df[subjects_no_transition_df['subject_id']!=best_candidate['subject_id'].item()]
subjects_pairs_df = pd.concat(selected_pairs)
# subjects_pairs_df['transition'] = 0

# %%
# Put both datasets together
group = np.arange(1,subjects_transition_df.shape[0]+1)
subjects_transition_df['group'] = group
subjects_pairs_df['group'] = group
joint_dataset_df = pd.concat([subjects_transition_df,subjects_pairs_df],axis=0)
#%%
# Remaining samples
remaining_test_df = subjects_no_transition_df

#%%
# Keep variables that show at least 5% variance
from sklearn.feature_selection import VarianceThreshold
cat_vars = ['subject_id','visit','group','transition','GENOTYPE','research_group','subject_date',
            'KEYMED','PTNLANG','DXMOTHET']
features = list(set(joint_dataset_df.columns)-set(cat_vars))
th = 0.01
var_thres=VarianceThreshold(threshold=th)
var_thres.fit(joint_dataset_df[features])
new_cols = var_thres.get_support()
keep_features = list(np.array(features)[new_cols])
keep_features.extend(['subject_id','visit','group','transition','GENOTYPE','research_group','subject_date',])
print(f'features={keep_features}')

#%%
joint_dataset_df = joint_dataset_df[keep_features]
joint_dataset_df.to_csv('data/joint_dataset.csv')
#%%
remaining_test_df = remaining_test_df[[x for x in keep_features if x!='group']]
remaining_test_df.to_csv('data/remaining_test.csv')

#%%
mri_joint_idx = list(set(joint_dataset_df['subject_id']).intersection(set(mri_df['PTID'])))
mri_joint_df = mri_df.set_index('PTID').loc[mri_joint_idx]
mri_joint_df = mri_joint_df[(mri_joint_df['VISCODE2']=='sc')|(mri_joint_df['VISCODE2']=='scmri')]
mri_joint_df = mri_joint_df.sort_values(by='EXAMDATE')
mri_joint_df = mri_joint_df[~mri_joint_df.index.duplicated(keep='first')]
mri_joint_df.to_csv('mri_joint_dataset.csv')
# %%

# Variables 
# 'research_group' - Research group first assigned to the subject
# 'GENOTYPE' - Apolipoprotein E (Apo-E) Genotype
# 2/2, 2/3, 2/4, 3/3, 3/4, 4/4
# 'study_duration' - months
# 'last_diagnosis' - 1:CN, 2:MCI, 3:Dementia
# 'DIAGNOSIS' - Initial diagnosis
#           1	CN
#           2	MCI
#           3	Dementia
# 'NPPDXI' - Contusion/traumatic brain injury of any type, Acute - NACC Neuropathology Data Form
#           0	No
#           1	Yes
#           8	Not assessed
#           9	Missing/unknown
# 'NPPDXJ' - Contusion/traumatic brain injury of any type, Chronic - NACC Neuropathology Data Form
#           0	No
#           1	Yes
#           8	Not assessed
#           9	Missing/unknown
# 'BCPREDX' - Pre-visit Diagnosis - Diagnostic Summary - Baseline Changes
#           1	NL
#           2	MCI
#           3	AD
# 'BCDEPRES' - Clinically relevant depression based on clinical judgement or GDS? - Diagnostic Summary - Baseline Changes
#           0	No
#           1	Yes
# 'GDTOTAL' - Geriatric Depression Scale (GDS) Total Scores - Geriatric Depression Scale
# 'NPIKTOT' - Neuropsychiatric Inventory (NPI) K. Sleep: Item score - Neuropsychiatric Inventory Examination
# 'PXHEART' - Physical Exam, Heart - Physical Exam
#           1	Normal
#           2	Abnormal
# 'MH12RENA' - Renal-Genitourinary - Medical History
#           0	No
#           1	Yes
# 'DXMPTR3' - Normal general cognitive function - Diagnostic Summary
#           0	No
#           1	Yes
#           2	Marginal
# 'RCT1408' - LDH - Laboratory Data
# 'RCT19' - Triglycerides (GPO) - Laboratory Data
# 'PTAU' - PTAU result - UPENN Longitudinal Biomarker Data (4 yr)
# 'TAU' - TAU result - UPENN Longitudinal Biomarker Data (4 yr)
# 'ABETA42' - ABETA42 result - UPENN Longitudinal Biomarker Data (4 yr)
# 'NPID' - Depression/Dysphoria - 0:No, 1:Yes - Neuropsychiatric Inventory Q
# 'NPIKSEV' - D. Depression/Dysphoria: Severity Ratings - Neuropsychiatric Inventory Q
#           1	1 - Mild (noticeable, but not a significant change).
#           2	2 - Moderate (significant, but not a dramatic change).
#           3	3 - Severe (very marked or prominent. A dramatic change).
# 'DILUTION_CORRECTED_CONC' - Dilution Corrected Concentration (pg/ml) - JANSSEN_PLASMA_P217_TAU
# 'CV' - Coefficient of Variation - JANSSEN_PLASMA_P217_TAU
# 'LDELTOTAL' - Logical Memory - (Delayed Recall) Total Number of Story Units Recalled - Neuropsychological Battery
# 'LIMMTOTAL' - Logical Memory - (Immediate Recall) Total Number of Story Units Recalled - Neuropsychological Battery
# 'FAQTOTAL' - Total Score - Functional Assessment Questionnaire
# 'TRAASCOR' - Trail Making Test Part A - Neuropsychological Battery
# 'TOTSCORE' - Total Score (ADAS 11) - ADAS-Cognitive
# 'TOTAL13' - Total Score (ADAS 13) - ADAS-Cognitive
# 'MMSCORE' - MMSE TOTAL SCORE - Mini Mental State Exam
# 'CDRSB' - CDR-SB - Clinical Dementia Rating
# 'KEYMED' - At this visit, is participant on any of the following medication? - Key Background Medications
#           0	None of the above
#           1	Aricept
#           2	Cognex
#           3	Exelon
#           4	Namenda
#           5	Razadyne
#           6	Anti-depressant medication
#           7	Other behavioral medication
# 'PTHOME' - Type of Participant residence - Participant Demographic Information
#           1   House (owned or rented), 
#           2   Condo/Co-op (owned)
#           3	Apartment (rented),     
#           4   Mobile Home, 
#           5   Retirement Community
#           6	Assisted Living, 7 Skilled Nursing Facility, 8 Other (specify)
#           9	House (rented), 10House (owned))
# 'PTWORK' - Primary occupation during most of adult life? - Participant Demographic Information
#           1	Professional and Higher Executive Occupations, Chief Executives
#           2	Middle Professional Occupations/ Small Business Owners
#           3	Managers
#           4	Support Personnel, Drafters, Technicians
#           5	Arts, Design, Entertainment, Sports Occupations – Non-Professional
#           6	Aides / Assistants / Clerks
#           7	Laborers
#           8	Other - Never Employed
#           9	Missing/Unknown
# 'PTMARRY' -  Participant Marital Status - Participant Demographic Information
#           1	Married
#           2	Widowed
#           3	Divorced
#           4	Never married
#           5	Unknown
#           6	Domestic Partnership
# 'PTNOTRT' - Participant Retired? - Participant Demographic Information
#           0   No 
#           1   Yes
#           2   Not Applicable
# 'PTCOGBEG' -  Year of onset of cognitive symptoms (best estimate) - Participant Demographic Information
# 'PTLANGTTL' - Total number of languages spoken (based on table above) - Participant Demographic Information
# 'PTSPOTTIM' - What percentage of the time during an average day do you speak another language? - Participant Demographic Information
# 'PTRTYR' - Retirement Date - Participant Demographic Information
# 'PTENGSPK' - Do you consider yourself a monolingual English speaker? - Participant Demographic Information
#           0   No
#           1   Yes
# 'VSBPSYS' - Systolic - mmHg - Vital Signs
# 'VSBPDIA' -  Diastolic - mmHg - Vital Signs
# 'HMT40' - Test HMT40; Hemoglobin - Laboratory Data
# 'AXDPMOOD' - Depressed mood (1:Absent, 2:Present) - Diagnosis and Symptoms Checklist
# 'DXNODEP' Mild Depression (1:Yes) - Diagnostic Summary
# 'NPIDTOT' - Depression/Dysphoria: Item score - Neuropsychiatric Inventory Examination
# 'NPID8' - Does {P} show any other signs of depression or sadness? - Neuropsychiatric Inventory Examination
#           0   No,
#           1   Yes
# 'NPIDSEV' - Depression/Dysphoria: Severity Ratings - Neuropsychiatric Inventory Q
#           1   Mild (noticeable, but not a significant change).
#           2   Moderate (significant, but not a dramatic change).
#           3   Severe (very marked or prominent. A dramatic change).
# 'BSXSYMNO' - Symptom Number - Documentation of Baseline Symptoms Log
#           1   Nausea
#           2   Vomiting
#           3   Diarrhea
#           4   Constipation
#           5   Abdominal discomfort
#           6   Sweating
#           7   Dizziness
#           8   Low energy
#           9   Drowsiness
#      	    10  Blurred vision
#      	    11  Headache
#      	    12  Dry mouth
#      	    13  Shortness of breath
#      	    14  Coughing
#      	    15  Palpitations
#      	    16  Chest pain
#      	    17  Urinary discomfort
#      	    18  Urinary frequency
#      	    19  Ankle swelling
#      	    20  Musculoskeletal pain
#      	    21  Rash
#      	    22  Insomnia
#      	    23  Depressed mood
#      	    24  Crying
#      	    25  Elevated mood
#      	    26  Wandering
#      	    27  Fall
#      	    28  Other
# 'BSXSEVER' - Severity - Documentation of Baseline Symptoms Log
#           1	Mild
#           2	Moderate
#           3	Severe
# 'BSXCHRON' - Chronicity - Documentation of Baseline Symptoms Log
#           1	Single occurrence
#           2	Intermittent
#           3	Persistent
# 'BCDPMOOD' - Depressed mood - Baseline Symptoms Checklist
#           1   Absent
#           2   Present 
# 'MHNUM' - System Number - Medical History
#          1   Psychiatric
#          2   Neurologic
#          3   Head, Eyes, Ears, Nose, Throat
#          4   Cardiovascular
#          5   Respiratory
#          6   Hepatic
#          7   Dermatologic-Connective Tissue
#          8   Musculoskeletal
#          9   Endocrine-Metabolic
#          10  Gastrointestinal
#          11  Hematopoietic-Lymphatic
#          12  Renal-Genitourinary
#          13  Allergies or Drug Sensitivities
#          14  Alcohol Abuse
#          15  Drug Abuse
#          16  Smoking
#          17  Malignancy
#          18  Major Surgical Procedures
#          19  Other
# 'BAT126' - Test BAT126; Vitamin B12 - Laboratory Data
# 'IES4' - I had trouble falling asleep, because of pictures or thoughts about it that came into my mind.
#          1	Not at All
#          2	Rarely
#          3	Sometimes
#          4	Often
# 'NPIK' - Sleep - Neuropsychiatric Inventory Examination
#           0   No
#           1   Yes
# 'NPIK1' - Does {P} have difficulty falling asleep? - Neuropsychiatric Inventory Examination
#           0   No
#           1   Yes
# 'NPIK2' - Does {P} get up during the night (do not count if the patient gets up once 
#           or twice per night only to go to the bathroom and falls back asleep immediately)? - Neuropsychiatric Inventory Examination
#           0   No
#           1   Yes
# 'NPIK7' - Does {P} sleep excessively during the day? - Neuropsychiatric Inventory Examination
#           0   No
#           1   Yes
# 'FAQSHOP' - Shopping alone for clothes, household necessities, or groceries - Functional Assessment Questionnaire
#          0	Normal (0)
#          1	Never did, but could do now (0)
#          2	Never did, would have difficulty now (1)
#          3	Has difficulty, but does by self (1)
#          4	Requires assistance (2)
#          5	Dependent (3)
# 'AXVISION' - Blurred vision - Diagnosis and Symptoms Checklist
#           1   Absent
#           2   Present
# 'BCVISION' - Blurred vision - Baseline Symptoms Checklist
#           1   Absent
#           2   Present
# 'BCHDACHE' - Headache - Baseline Symptoms Checklist
#           1   Absent
#           2   Present
# 'AXHDACHE' - Headache - Diagnosis and Symptoms Checklist
#           1   Absent
#           2   Present
# 'AXINSOMN' - Insomnia - Diagnosis and Symptoms Checklist
#           1   Absent
#           2   Present
# 'BCINSOMN' - Insomnia - Baseline Symptoms Checklist
#           1   Absent
#           2   Present
# 'AXFALL' - Fall - Diagnosis and Symptoms Checklist
#           1   Absent
#           2   Present
# 'BCFALL' Fall - Baseline Symptoms Checklist
#           1   Absent
#           2   Present
# 'AXMUSCLE' - Muscloskeletal pain - Diagnosis and Symptoms Checklist
#           1   Absent
#           2   Present
# 'BCMUSCLE' - Muscloskeletal pain - Baseline Symptoms Checklist
#           1   Absent
#           2   Present
# 'AXCHEST' - Chest pain - Diagnosis and Symptoms Checklist
#           1   Absent
#           2   Present
# 'BCCHEST' - Chest pain - Baseline Symptoms Checklist
#           1   Absent
#           2   Present
# 'RCT392' - Creatinine (Rate Blanked) - Laboratory Data
# 'RCT14' - Creatine Kinase - Laboratory Data
# 'BCSTROKE' - Did subject have a stroke? - Baseline Symptoms Checklist
#           0   No
#           1   Yes
# 'HMSTROKE' - History of Stroke - Modified Hachinski
#           0   Absent,
#           2   Present - 2 points
# 'HMSOMATC' - Somatic Complaints - Modified Hachinski
#           0   Absent,
#           1   Present - 1 points
# 'HMSCORE' - TOTAL SCORE - Modified Hachinski
# 'MH4CARD' - Cardiovascular (0:No, 1:Yes) - Medical History
# 'PXPERIPH' - Peripheral Vascular - Physical Exam
#           1   Normal
#           2   Abnormal
# 'FAQTRAVL' - Traveling out of the neighborhood, driving, or arranging to take public transportation - Functional Assessment Questionnaire
#          0	Normal (0)
#          1	Never did, but could do now (0)
#          2	Never did, would have difficulty now (1)
#          3	Has difficulty, but does by self (1)
#          4	Requires assistance (2)
#          5	Dependent (3)
# 'FAQGAME' - Playing a game of skill such as bridge or chess, working on a hobby - Functional Assessment Questionnaire
#          0	Normal (0)
#          1	Never did, but could do now (0)
#          2	Never did, would have difficulty now (1)
#          3	Has difficulty, but does by self (1)
#          4	Requires assistance (2)
#          5	Dependent (3)
# 'FAQFINAN' - Writing checks, paying bills, or balancing checkbook - Functional Assessment Questionnaire
#          0	Normal (0)
#          1	Never did, but could do now (0)
#          2	Never did, would have difficulty now (1)
#          3	Has difficulty, but does by self (1)
#          4	Requires assistance (2)
#          5	Dependent (3)
# 'NXGAIT' - Gait - Neurological Exam
#           1   Normal,
#           2   Abnormal 
# 'CDHOME' - Home and Hobbies Score - Clinical Dementia Rating
# 'CDCARE' - Personal Care Score - Clinical Dementia Rating
# 'PTPLANG' - What is the participant's primary language? - Participant Demographic Information
#          1	English
#          2	Spanish
#          3	Other
# 'PTNLANG' - What is your native language? - Participant Demographic Information
#          1	English
#          2	Spanish
#          3	Other

# 1=Fronto-temporal Dementia, 2=Parkinson's Disease, 3=Huntington's Disease, 4=Progressive Supranuclear Palsy, 5=Alcohol-related Dementia, 6=NPH, 7=Major Depression, 8=Corticobasal Degeneration, 9=Vascular Dementia, 10=Prion-Associated Dementia, 11=HIV, 12=Primary Progressive Aphasia, 13=Posterior Cortical Dysfunction, 14=Other (specify)
# 1=Mild, 2=Moderate, 3=Severe

# 1=Occasionally, 2=Often, 3=Frequently, 4=Very frequently
# 1=Mild, 2=Moderate, 3=Marked
# 0=Not at all; 1=Minimally; 2=Mildly; 3=Moderately; 4=Severely; 5=Very severely or extremely
#%%
# Change weight and height to metric system
# joint_dataset_df['HEIGHT'] = joint_dataset_df['VSHEIGHT'].where(joint_dataset_df['VSHTUNIT']==2,joint_dataset_df['VSHEIGHT']*2.54)
# joint_dataset_df['WEIGHT'] = joint_dataset_df['VSWEIGHT'].where(joint_dataset_df['VSWTUNIT']==2,joint_dataset_df['VSWEIGHT']*0.453592)
# Calculate BMI
# joint_dataset_df['BMI'] = joint_dataset_df['WEIGHT']/((joint_dataset_df['HEIGHT']/100)**2)
#%%
# id_df = pd.DataFrame(joint_dataset_df['subject_id'].str.split('_').to_list(),columns=['SITEID','C','ID'], index=joint_dataset_df.index)[['SITEID','ID']].astype(int)
# joint_dataset_df = pd.merge(left=id_df,right=joint_dataset_df,left_index=True,right_index=True)
# joint_dataset_df = joint_dataset_df.drop('subject_id',axis='columns')
# joint_dataset_df = joint_dataset_df.set_index(['SITEID','ID'])
# %%
# joint_dataset_df.to_csv('joint_dataset.csv')
# 1=Uncertain, 2=Mildly Confident, 3=Moderately Confident, 4=Highly Confident
# %%
# Load data from ADNI dataset
# /workspaces/R21/src/march2025/ADNI_PICSLASHS_20Jun2025.csv
# /workspaces/R21/src/march2025/UCD_WMH_20Jun2025.csv
# /workspaces/R21/src/march2025/UCSFFSX7_20Jun2025.csv
mri_df = pd.read_csv('UCSFFSX7_20Jun2025.csv')
mri_id_df = pd.DataFrame(mri_df['PTID'].str.split('_').to_list(),columns=['SITEID','C','ID'], index=mri_df.index)[['SITEID','ID']].astype(int)
mri_df = pd.merge(left=mri_id_df,right=mri_df,left_index=True,right_index=True)
mri_df = mri_df.set_index(['PTID'])
# %%
mri_filtered_df = mri_df[mri_df.index.isin(joint_dataset_df.index)]
mri_filtered_df = mri_filtered_df[mri_filtered_df['VISCODE'].isin(['bl','sc','4_init'])]

# %%
# /workspaces/R21/src/march2025/ADNIMERGE_22Jun2025.csv
adni_df = pd.read_csv('ADNIMERGE_22Jun2025.csv')
adni_cols_bl = list(adni_df.columns[['bl' in x for x in adni_df.columns]])
adni_cols_bl.extend(['PTID','RID','SITE','DX','APOE4','PTGENDER'])
adni_bl_df = adni_df[adni_cols_bl]
# %%
no_change_condition = ((adni_df['DX_bl']=='CN') & (adni_df['DX']=='CN'))
adni_no_change = adni_bl_df[no_change_condition]
adni_no_change_bl = adni_no_change.sort_values(by='Years_bl').drop_duplicates(subset=['RID'], keep='first')
# %%
change_condition = ((adni_df['DX_bl']=='CN') & ((adni_df['DX']=='MCI')|(adni_df['DX']=='Dementia')))
adni_change = adni_bl_df[change_condition]
adni_change_bl = adni_change.sort_values(by='Years_bl').drop_duplicates(subset=['RID'], keep='first')
#%%
# Remove subjects who transitioned from the set of subjects who remained CN
adni_no_change_bl = adni_no_change_bl[~(adni_no_change_bl['RID'].isin(adni_change_bl['RID']))]
#%%
modifiable_factors_list = []
drop_cols_ = ['visit','research_group','DIAGNOSIS','MRI_acquired']
#%%
data_sc_bl_df = data_df[(data_df['visit']=='sc')|(data_df['visit']=='bl')]
# Strings are found in RCT1408, RCT19, and RCT20
data_sc_bl_df.loc[:,'RCT1408'] = pd.to_numeric(data_sc_bl_df.loc[:,'RCT1408'], errors='coerce')
data_sc_bl_df.loc[:,'RCT19'] = pd.to_numeric(data_sc_bl_df.loc[:,'RCT19'], errors='coerce')
data_sc_bl_df.loc[:,'RCT20'] = pd.to_numeric(data_sc_bl_df.loc[:,'RCT20'], errors='coerce')
data_sc_bl_df = data_sc_bl_df.astype(datatypes_dict)
# Drop data_sc_bl_df['BCDEPRES']
data_sc_bl_df = data_sc_bl_df.drop(['BCDEPRES'],axis='columns')
data_cols = data_sc_bl_df.columns[data_sc_bl_df.notna().sum(axis=0)>0]
data_sc_bl_df = data_sc_bl_df[data_cols]
#%%
for adni_df_ in [adni_no_change_bl,adni_change_bl]:
    # adni_df_ = adni_no_change_bl
    modifiable_factors_df = data_sc_bl_df.set_index('subject_id').loc[adni_df_['PTID']]
    # modifiable_factors_df = modifiable_factors_df.astype(datatypes_dict)
    modifiable_factors_bl_df = modifiable_factors_df[modifiable_factors_df['visit']=='bl']
    modifiable_factors_bl_df = modifiable_factors_bl_df.drop(drop_cols_,axis='columns')
    # bl_cols = modifiable_factors_bl_df.columns[modifiable_factors_bl_df.notna().sum(axis=0)>0]
    # modifiable_factors_bl_df = modifiable_factors_bl_df[bl_cols]
    modifiable_factors_sc_df = modifiable_factors_df[modifiable_factors_df['visit']=='sc']
    modifiable_factors_sc_df = modifiable_factors_sc_df.drop(drop_cols_,axis='columns')
    # sc_cols = modifiable_factors_sc_df.columns[modifiable_factors_sc_df.notna().sum(axis=0)>0]
    # modifiable_factors_sc_df = modifiable_factors_sc_df[sc_cols]
    sc_bl_cols = list(set(modifiable_factors_sc_df.columns).intersection(modifiable_factors_bl_df.columns))

    # Replacing missing weight info in SC tp using BL tp
    idx_na_weight = modifiable_factors_sc_df['VSWEIGHT'].isna()
    n_replaced_weight = modifiable_factors_bl_df[idx_na_weight]['VSWEIGHT'].notna().sum()
    if n_replaced_weight>0:
        modifiable_factors_sc_df.loc[idx_na_weight,['VSWEIGHT','VSWTUNIT']] = modifiable_factors_bl_df.loc[idx_na_weight,['VSWEIGHT','VSWTUNIT']]
        print(f'N={n_replaced_weight} fields for VSWEIGHT were missing and replaced using baseline timepoint information' ) 

    # Replacing missing height info in SC tp using m60
    idx_na_height = modifiable_factors_sc_df['VSHEIGHT'].isna()
    replaced_height_df = modifiable_factors_df.loc[idx_na_height.index[idx_na_height]].sort_values(by=['VSHEIGHT'],ascending=False).reset_index().drop_duplicates(subset=['subject_id']).set_index('subject_id')
    replaced_height_df = replaced_height_df[replaced_height_df['VSHEIGHT'].notna()]
    n_replaced_height = replaced_height_df.shape[0] 
    if n_replaced_height>0:
        modifiable_factors_sc_df.loc[replaced_height_df.index,['VSHEIGHT','VSHTUNIT']] = replaced_height_df.loc[replaced_height_df.index,['VSHEIGHT','VSHTUNIT']]
        print(f'N={n_replaced_height} fields for VSHEIGHT were missing and replaced using other timepoint information' )

    # Weight in Kgs
    modifiable_factors_sc_df['VSWEIGHT(kg)'] = modifiable_factors_sc_df['VSWEIGHT'].where(modifiable_factors_sc_df['VSWTUNIT']==2,modifiable_factors_sc_df['VSWEIGHT']*0.453592)
    modifiable_factors_bl_df['VSWEIGHT(kg)'] = modifiable_factors_bl_df['VSWEIGHT'].where(modifiable_factors_bl_df['VSWTUNIT']==2,modifiable_factors_bl_df['VSWEIGHT']*0.453592)
    # Height in cm
    modifiable_factors_sc_df['VSHEIGHT(cm)'] = modifiable_factors_sc_df['VSHEIGHT'].where(modifiable_factors_sc_df['VSHTUNIT']==2,modifiable_factors_sc_df['VSHEIGHT']*2.54)
    modifiable_factors_sc_df['BMI'] = modifiable_factors_sc_df['VSWEIGHT(kg)']/((modifiable_factors_sc_df['VSHEIGHT(cm)']/100)**2)
    # weight_ = (modifiable_factors_sc_df['VSWEIGHT_kg']+modifiable_factors_bl_df['VSWEIGHT_kg'])/2

    # Drop repeated variables
    # modifiable_factors_sc_df = modifiable_factors_sc_df.drop(sc_bl_cols,axis='columns')
    modifiable_factors_sc_df = modifiable_factors_sc_df.drop(['VSHEIGHT','VSHTUNIT'],axis='columns')
    # modifiable_factors_bl_df = modifiable_factors_bl_df.drop(sc_bl_cols,axis='columns')

    # Add suffixes
    modifiable_factors_sc_df = modifiable_factors_sc_df.add_suffix('_sc')
    modifiable_factors_bl_df = modifiable_factors_bl_df.add_suffix('_bl')
    modifiable_factors_ = pd.merge(left=modifiable_factors_sc_df, right=modifiable_factors_bl_df, right_index=True, left_index=True)
    modifiable_factors_ = modifiable_factors_.drop(['PTGENDER_sc','GENOTYPE_sc','subject_age_sc'],axis='columns').dropna(axis='columns',how='all')
    adni_df_ = adni_df_.set_index('PTID')
    modifiable_factors_ = pd.merge(left=modifiable_factors_,right=adni_df_, right_index=True, left_index=True)
    modifiable_factors_list.append(modifiable_factors_)

# %%
# Look for subjects in the 'no transition group' who have similar attributes to the ones in the 'transition group'
selected_pairs = []
num_vars_names = ['subject_age_bl']
cat_vars_names = ['PTGENDER_bl','GENOTYPE_bl']
subjects_no_transition_df = modifiable_factors_list[0] 
subjects_transition_df = modifiable_factors_list[1]
subjects_id_transition = subjects_transition_df.index
#%%
for subject in subjects_id_transition:
    
    # Check age, sex, and duration of study, and rank candidates based on the norm of the difference
    num_vars_transition = subjects_transition_df.loc[subject][num_vars_names]
    cat_vars_transition = subjects_transition_df.loc[subject][cat_vars_names]
    diff_df = (subjects_no_transition_df[num_vars_names]-num_vars_transition.values).astype(float)
    diff_df['norm'] = np.linalg.norm(diff_df,axis=1)
    diff_df = diff_df.sort_values(by=['norm'], ascending=True)
    diff_df = pd.merge(left=diff_df, right=subjects_no_transition_df[cat_vars_names], left_index=True, right_index=True)
    cat_vars_sim_idx = np.all(diff_df[cat_vars_names]==cat_vars_transition.values, axis=1)
    canditates_df = diff_df.loc[cat_vars_sim_idx]
    best_candidate = subjects_no_transition_df.loc[[canditates_df.index[0]]]
    selected_pairs.append(best_candidate)
    # Remove subject from the pool
    # subjects_no_transition_df = subjects_no_transition_df[subjects_no_transition_df['subject_id']!=best_candidate['subject_id'].item()]
    subjects_no_transition_df = subjects_no_transition_df.drop(best_candidate.index,axis='rows')
subjects_pairs_df = pd.concat(selected_pairs)
subjects_pairs_df['transition'] = 0
subjects_transition_df['transition'] = 1
# %%
# Put both datasets together
group = np.arange(1,subjects_transition_df.shape[0]+1)
subjects_transition_df['group'] = group
subjects_pairs_df['group'] = group
joint_dataset_df = pd.concat([subjects_transition_df,subjects_pairs_df],axis=0)
# %%
joint_dataset_df.to_csv('joint_dataset.csv')
# %%
#%%
from sklearn.feature_selection import VarianceThreshold
cat_vars = ['SITE','RID','group','transition','GENOTYPE_bl','ABETA_bl',
            'PTGENDER','DX','DX_bl','EXAMDATE_bl','FSVERSION_bl','FLDSTRENG_bl']
features = list(set(joint_dataset_df.columns)-set(cat_vars))
th = 0.05
var_thres=VarianceThreshold(threshold=th)
var_thres.fit(joint_dataset_df[features])
new_cols = var_thres.get_support()
print(np.array(features)[new_cols])
# %%
