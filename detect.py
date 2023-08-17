import os
import cv2 
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from model.loss import CrossEntropyLabelSmooth
#from model import convnext
from model import transformer
from dataset_muti import MyDataset
from urllib.request import urlopen

csv_save_filename="ViT_final_A.csv"
modelName = './pth/Tiny/ViT_final_A.pth'
print(modelName)
detect_dir = 'data/detect/'
result_dir = 'result/'
num_classes1 = 2
num_classes2 = 4
num_classes3 = 14
target_name = {
'name1':['TFT','CF'],
'name2': ['NP','UP','OP','INT'],
'name3': ['CF REPAIR FAIL','PI SPOT-WITH PAR','POLYMER','GLASS BROKEN','PV-HOLE-T','CF DEFECT','CF PS DEFORMATION','FIBER','AS-RESIDUE-E','LIGHT METAL','GLASS CULLET','ITO-RESIDUE-T','M1-ABNORMAL','ESD']
}

def makedirs(path):
    try:
        os.makedirs(path)
    except:
        return


device = torch.device("cuda:0" if torch.cuda.is_available() and 'gpu' == 'gpu' else 'cpu')

'''mdoelsave.pth location'''


''' Mode read'''
## for a b 
model = transformer.Transformer(num_class1=num_classes1, num_class2=num_classes2, num_class3=num_classes3).to(device)

##　for c
#model = transformer.Transformer_sharefeature(num_class1=num_classes1, num_class2=num_classes2, num_class3=num_classes3).to(device)

#model = model.to(device)
model.load_state_dict(torch.load(modelName))
model.eval()


'''data - loader'''
batch_size=1
epoch = 1
datadic={}
coarse_labels,fine_labels,third_labels = target_name['name1'],target_name['name2'],target_name['name3']

detect_dataset = MyDataset(detect_dir, 'pred',target_name)
detect_generator = DataLoader(dataset=detect_dataset, batch_size=1, shuffle=False)

datalen = len(os.listdir(detect_dir))

dfsave = pd.DataFrame()

r=0
#for e in range(epoch):
for j, (img, img_path) in enumerate(tqdm(detect_generator)):
        if j < datalen :
                print("-----------Number-----:"+str(j))
                #----for test.csv testing----
                batch_x ,imgpath = img.to(device),img_path
                print(imgpath)

        #                print(imgpath)
                ''' Tensor balue'''
                superclass_pred,subclass_pred ,subtwoclass_pred= model(batch_x) 
                #predicted_super = torch.argmax(superclass_pred, dim=1)#tensor([1])
                #predicted_sub = torch.argmax(subclass_pred, dim=1)#tensor([9])
                #predicted_sub tow= torch.argmax(subtwoclass_pred, dim=1)#tensor([9])

                ''' confidence  & classes'''
                ''' - superclasses'''
                probs_super = torch.nn.functional.softmax(superclass_pred, dim=1) 
                super_value,super_index=torch.topk(probs_super,k=2,largest=True) #torch.topk(取出前幾大) , 2取出幾個        
                conf,classes = torch.max(probs_super,1) 
                imgclass= coarse_labels[(classes.item())]
                print('superclass',conf,imgclass)

                ''' - subclasses'''
                probs_sub = torch.nn.functional.softmax(subclass_pred, dim=1)
                sub_value,sub_index=torch.topk(probs_sub,k=4,largest=True)#torch.topk(取出前幾大) , 2取出幾個        
                conf_sub,classes_sub = torch.max(probs_sub,1)
                imgclass_sub= fine_labels[(classes_sub.item())]
                print('subclass',conf_sub,imgclass_sub)

                ''' - subtwoclasses'''
                probs_subtwo = torch.nn.functional.softmax(subtwoclass_pred, dim=1)
                subtwo_value,subtwo_index=torch.topk(probs_subtwo,k=5,largest=True)#torch.topk(取出前幾大) , 2取出幾個        
                conftwo_sub,classestwo_sub = torch.max(probs_subtwo,1)
                imgclasstwo_sub= third_labels[(classestwo_sub.item())]
                print('subclass',conftwo_sub,imgclasstwo_sub)
                
                imagename = imgpath[0].split('/')[2]

                ''' Get into datadic '''
                output_dic = {
                        'super_conf':[str(index)[:6] for index in super_value[0].tolist()],
                        'super_class':[coarse_labels[index] for index in super_index[0].tolist()],
                        'sub_conf':[str(index)[:6] for index in sub_value[0].tolist()],
                        'sub_class':[fine_labels[index] for index in sub_index[0].tolist()],
                        'subtwo_conf':[str(index)[:6] for index in subtwo_value[0].tolist()],
                        'subtwo_class':[third_labels[index] for index in subtwo_index[0].tolist()],
                        'Layer_1_ans':imgclass,
                        'Layer_1_conf':str(conf[0].tolist())[:6],
                        'Layer_2_ans':imgclass_sub,
                        'Layer_2_conf':str(conftwo_sub[0].tolist())[:6],
                        'Layer_3_ans':imgclasstwo_sub,
                        'Layer_3_conf':str(conf_sub[0].tolist())[:6],
                        'Layer_1_True':imagename.split('@')[2],
                        'Layer_2_True':imagename.split('@')[1],
                        'Layer_3_True':imagename.split('@')[0]
                }
                ''' dataframe concat'''
                datadic[imagename] = output_dic
                df = pd.DataFrame(datadic)
                df = df.T

                if  len(dfsave) == 0 :
                        dfsave = df 
                else :
                        dfsave = pd.concat([df,dfsave],axis=0)

print("-----------Number-----:"+str(j))
'''datasave cleaner'''
index_duplicates = dfsave.index.duplicated()
dfsave = dfsave.loc[~index_duplicates]
#dfsave.reset_index(drop=True,inplace=True)

makedirs(result_dir)
dfsave.to_csv(result_dir+csv_save_filename,index=True,index_label='ImagePath')
print('data_save:'+result_dir+csv_save_filename)
exit()

