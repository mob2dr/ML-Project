import preprocessing as pre 
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.impute import KNNImputer
from sklearn.utils import class_weight


seleted_cols = ['fallnr', 'log_euroscore_perc', 'height_cm', 'eks',
               'nora_max_mcgperml', 'icu_max_postop_plusbilanz_ml',
               'gesamtbilanz_l', 'bilanz_pro_kg_lkg', 'attest_ja1',
               'haematokrit_preop_percent', 'tropi_preop_ngperml',
               'tropi_max_postop_ngperml', 'tropi_min_postop_ngperml',
               'tropt_min_postop_ngperl', 'ckmb_preop_mcgperl',
               'ck_max_postop_uperl', 'ck_min_postop_uperl', 'age_at_surg',
               'delir_dauer_tage','therapie_relevantes_delir_ja1']

data = pre.read_data_from_file(path = 'Binary_Classification_Medical.txt', delimiter = '\t')
data.dropna(subset=['therapie_relevantes_delir_ja1'],axis=0, inplace= True)
data = data[seleted_cols].copy()

# Specify the columns for each transformation
columns_boxcox = ['log_euroscore_perc', 'eks','nora_max_mcgperml','icu_max_postop_plusbilanz_ml',
                 'gesamtbilanz_l','bilanz_pro_kg_lkg','tropi_preop_ngperml','tropi_min_postop_ngperml',
                 'tropt_min_postop_ngperl', 'ckmb_preop_mcgperl','ck_max_postop_uperl',
                 'ck_min_postop_uperl']

columns_onehot_encode = ['delir_dauer_tage']
columns_knn_impute = ['haematokrit_preop_percent', 'haematokrit_preop_percent', 'tropi_max_postop_ngperml',
                     'tropi_min_postop_ngperml','tropt_min_postop_ngperl','ckmb_preop_mcgperl']

# Create the pipeline steps
transformer = ColumnTransformer(
    transformers=[
        ('boxcox_transform', PowerTransformer(method='box-cox'), columns_boxcox),
       
    ],
    remainder='passthrough'
)

# Define the pipeline
pipeline = Pipeline(
    steps=[
        ('preprocess', transformer),
        ('knn_impute', KNNImputer(n_neighbors=5)),
        ('xgboost_model', xgb.XGBClassifier())
    ]
)

X = data.drop(['therapie_relevantes_delir_ja1'],axis=1)
y = data['therapie_relevantes_delir_ja1']
# Fit the pipeline on the training data
classes_weights = class_weight.compute_sample_weight(
    class_weight='balanced',
    y=y
)

# Fit the pipeline on the training data
pipeline.fit(X , y, xgboost_model__sample_weight=classes_weights)

# Make predictions using the XGBoost model
y_pred = pipeline.predict(X)

print('classification_report on training')
print(classification_report(y, y_pred))





