import os
import sys
import six
import pandas as pd
from radiomics import featureextractor
sys.path.append('../../4D_radiomics/')
from data_loader.image_loader import get_id_label
from data_loader.paths_and_dirs import get_filepath
from data_loader.image_loader import read_nifti, get_subject_sequence



# Set path to nifti files
data_dir = '/media/mehdi/KTH/DeepLearning/DataRepository/4D_Breast_MRI/BreastCancer4DMRI/NewBatch/'
img_paths = get_filepath(data_dir, 'cropped')
mask_paths = get_filepath(data_dir, 'mask')


# Set writing path
filename_xlsx = 'batch2_new_Larger.xlsx'
feature_file = os.path.join('../features/', filename_xlsx)



# Instantiating Radiomics Feature Extraction
extractor = featureextractor.RadiomicsFeatureExtractor()
param_path = os.path.join(os.getcwd(), 'params.yaml')
extractor = featureextractor.RadiomicsFeatureExtractor(param_path)
print('Extraction parameters:\n\t', extractor.settings)
print('Enabled filters:\n\t', extractor.enabledImagetypes)
print('Enabled features:\n\t', extractor.enabledFeatures) 
    

data_len = len(img_paths)

for ind in range(data_len):
    
    data_path = img_paths[ind]
    data_mask_path = mask_paths[ind]
    
    subject_id, subject_label =  get_id_label(data_path)
    print('\n\n\n\n\n\n\n\n\n\n\n\n\n')
    print('working on case {} out of {}:'.format(ind, data_len))
    print('\n\n\n\n\n\n\n\n\n\n\n\n\n')
    
    img_itk, img_size, img_spacing, img_origin, img_direction = read_nifti(data_path) 
    mask_itk, mask_size, mask_spacing, mask_origin, mask_direction = read_nifti(data_mask_path)   
    
    img_seq = get_subject_sequence(img_itk, img_size, img_spacing, img_origin, mask_direction)   
    
    for num, name in enumerate(img_seq):
        
        features = extractor.execute(name, mask_itk)
        
        features_vector = {}
        for key, value in six.iteritems(features):
            if key.startswith('original') or key.startswith('wavelet') or \
            key.startswith('log'):
                features_vector['Subject_ID'] = subject_id
                features_vector['Subject_Label'] = subject_label
                features_vector[key] = features[key]
                
                
        if ind ==0:
            
            if num == 0:
                df1 = pd.DataFrame(data=features_vector, index=[ind])
            elif num == 1:
                df2 = pd.DataFrame(data=features_vector, index=[ind])
            elif num == 2:
                df3 = pd.DataFrame(data=features_vector, index=[ind])
            elif num == 3:
                df4 = pd.DataFrame(data=features_vector, index=[ind])
            elif num == 4:
                df5 = pd.DataFrame(data=features_vector, index=[ind])
            elif num == 5:
                df6 = pd.DataFrame(data=features_vector, index=[ind])
                
        elif ind !=0:
            
            if num == 0:
                df1 = df1.append(pd.DataFrame(data=features_vector, index=[ind]))
            elif num ==1:
                 df2 = df2.append(pd.DataFrame(data=features_vector, index=[ind]))
            elif num ==2:
                 df3 = df3.append(pd.DataFrame(data=features_vector, index=[ind]))
            elif num ==3:
                 df4 = df4.append(pd.DataFrame(data=features_vector, index=[ind]))
            elif num ==4:
                 df5 = df5.append(pd.DataFrame(data=features_vector, index=[ind]))
            elif num ==5:
                 df6 = df6.append(pd.DataFrame(data=features_vector, index=[ind]))


writer = pd.ExcelWriter(feature_file, engine='xlsxwriter')
df1.to_excel(writer, sheet_name='Sequence1')
df2.to_excel(writer, sheet_name='Sequence2')
df3.to_excel(writer, sheet_name='Sequence3')
df4.to_excel(writer, sheet_name='Sequence4')
df5.to_excel(writer, sheet_name='Sequence5')
df6.to_excel(writer, sheet_name='Sequence6')
writer.save()



