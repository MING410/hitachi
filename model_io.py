import os
import torch

def load_model_by_path(model_save_path, model, gpu_id = None):
    if not os.path.exists(model_save_path): return
    loc = 'cpu' if gpu_id is None else 'cuda:{}'.format(gpu_id)
    pretrained_dict = torch.load(model_save_path, map_location=loc)
    if 'state_dict' in pretrained_dict: pretrained_dict = pretrained_dict['state_dict']
    model_dict = model.state_dict()
    # pretrained_dict.pop('netG.model.1.weight', '404')
    pretrained_dict_new = {k: v for k, v in pretrained_dict.items()
                       if (k in model_dict and model_dict[k].data.shape == v.data.shape)}
    # if 'netG.model.1.weight' not in pretrained_dict_new and 'netG.model.1.weight' in model_dict:
    #     pretrained_dict_new['netG.model.1.weight'] = model_dict['netG.model.1.weight']
    #     pretrained_dict_new['netG.model.1.weight'][:,:12] = pretrained_dict['netG.model.1.weight']

    model_dict.update(pretrained_dict_new)
    model.load_state_dict(model_dict)
    print("load model: ", model_save_path)

def save_model_by_path(model_save_path, model):
    save_dir, _ = os.path.split(model_save_path)
    if not os.path.isdir(save_dir): os.makedirs(save_dir)
    model_dic={k.replace('.module', ''):v for k,v in model.state_dict().items()}
    if torch.__version__ >= '1.6.0':
        print("(after torch1.6, model_save_path: ", model_save_path)
        torch.save(model_dic, model_save_path, _use_new_zipfile_serialization=False)
    else:
        print("before torch1.6, model_save_path: ", model_save_path)
        torch.save(model_dic, model_save_path)
