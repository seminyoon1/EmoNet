class_names = ['angry', 'happy', 'relaxed', 'sad']
num_classes = 4

TXT_PATH='/home/afromero/datos2/EmoNet/data'

def update_folder(config, folder):
  import os
  config.log_path = os.path.join(config.log_path, folder)
  config.model_save_path = os.path.join(config.model_save_path, folder)
  config.result_save_path = os.path.join(config.result_save_path, folder)

def update_config(config):
    import os, glob, math, imageio

    folder_parameters = os.path.join(config.dataset, config.mode_data, 'fold_'+config.fold, config.finetuning)
    update_folder(config, folder_parameters)
    if config.BLUR: update_folder(config, 'BLUR')
    if config.GRAY: update_folder(config, 'GRAY')
    
    # Remove 'Faces_256' from metadata path
    config.metadata_path = os.path.join(config.metadata_path)

    if config.pretrained_model=='':
        try:
            config.pretrained_model = sorted(glob.glob(os.path.join(config.model_save_path, '*.pth')))[-1]
            config.pretrained_model = os.path.basename(config.pretrained_model).split('.')[0]
        except:
            pass

    if config.test_model=='':
        try:
            config.test_model = sorted(glob.glob(os.path.join(config.model_save_path, '*.pth')))[-1]
            config.test_model = os.path.basename(config.test_model).split('.')[0]
        except:
            config.test_model = ''  

    return config