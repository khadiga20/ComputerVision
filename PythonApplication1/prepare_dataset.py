import yaml
import os

def create_data_yaml(path_to_classes_txt, path_to_data_yaml):
    if not os.path.exists(path_to_classes_txt):
        print(f'classes.txt file not found! Please create it at {path_to_classes_txt}')
        return

    with open(path_to_classes_txt, 'r') as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    number_of_classes = len(classes)

    data = {
        'path': r'C:\Users\Khadiga Yahia\.kaggle\PythonApplication1\cctv_weapon_data\Dataset\images',
        'train': 'train',
        'val': 'val',
        'nc': number_of_classes,
        'names': classes
    }

    with open(path_to_data_yaml, 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print(f'Created config file at {path_to_data_yaml}')

# تعديل المسارات حسب جهازك
path_to_classes_txt = r'C:\Users\Khadiga Yahia\.kaggle\PythonApplication1\cctv_weapon_data\Dataset\classes.txt'
path_to_data_yaml = r'C:\Users\Khadiga Yahia\.kaggle\PythonApplication1\cctv_weapon_data\Dataset\data.yaml'

create_data_yaml(path_to_classes_txt, path_to_data_yaml)

print('\nFile contents of data.yaml:\n')
with open(path_to_data_yaml, 'r') as f:
    print(f.read())
