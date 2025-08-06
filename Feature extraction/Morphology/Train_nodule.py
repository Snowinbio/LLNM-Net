import os
import pandas as pd
import torch
from skimage import io, transform, color, img_as_ubyte
from torch.utils.data import DataLoader
from model.Texture import TextureImageDataset, get_texture_transform
from model.Edge import EdgeImageDataset, get_edge_transform
from model.ResNet import get_resnet_model, train_model, evaluate_model, load_model
from model.Echogenicity import Echogenicity
from model.EdgeDetection import Detection, EdgeFeatureExtractor
from model.Shape import ShapeFeatureExtractor
from model.Classfication import LogisticRegressionWithANOVA
from model.Config import Config
import model.EdgeProcess as EdgeProcess
import model.EdgeDetectionProcess as EdgeDetectionProcess

import numpy as np
import cv2


def compute(img, min_percentile, max_percentile):
    max_percentile_pixel = np.percentile(img, max_percentile)
    min_percentile_pixel = np.percentile(img, min_percentile)
    return max_percentile_pixel, min_percentile_pixel

def aug(src):
    max_percentile_pixel, min_percentile_pixel = compute(src, 1, 99)
    src[src>=max_percentile_pixel] = max_percentile_pixel
    src[src<=min_percentile_pixel] = min_percentile_pixel
    out = np.zeros(src.shape, src.dtype)
    cv2.normalize(src, out, 255*0.1,255*0.9,cv2.NORM_MINMAX)
    return out

def get_lightness(src):
    hsv_image = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    lightness = hsv_image[:,:,2].mean()
    return  lightness

def process_pic(jpg_dir_path, nodule_dir_path, process_dir_path):
    length_value=5
    out_size=299
    jpgs_num = len(os.listdir(nodule_dir_path))     
    jpg_num = 0                                      
    for jpg_path in os.listdir(nodule_dir_path):
        jpg_num += 1
    
        # Anchoring and segmenting nodule regions
        origin_path=nodule_dir_path+"\\"+jpg_path
        nodule_label=io.imread(origin_path)
        nodule_label=np.array(nodule_label,copy=False)
        if len(nodule_label.shape)!=2:
            nodule_label=color.rgb2gray()
        nodule_label = transform.resize(image=nodule_label, output_shape=(out_size,out_size)) * 255
        nodule_label = np.where(nodule_label>100,255,0)
    
        get_true = False
        h_list,v_list = [],[]
        for i in range(nodule_label.shape[0]):
            for j in range(nodule_label.shape[1]):
                if nodule_label[i][j]>0:
                    h_list.append(i)
                    v_list.append(j)
                    get_true = True
        if not get_true:
            continue
        if min(h_list)-length_value<0:
            min_h_list = 0
        else:
            min_h_list = min(h_list)-length_value
        if min(v_list)-length_value<0:
            min_v_list = 0
        else:
            min_v_list = min(v_list)-length_value
        if max(h_list)+length_value>nodule_label.shape[0]-1:
            max_h_list = nodule_label.shape[0]-1
        else:
            max_h_list = max(h_list)+length_value
        if max(v_list)+length_value>nodule_label.shape[1]-1:
            max_v_list = nodule_label.shape[1]-1
        else:
            max_v_list = max(v_list)+length_value
            
        # Read and segment original images
        jpg_frame_img = np.zeros((out_size,out_size))
        jpg_name_path=jpg_dir_path+"\\"+jpg_path
        jpg_img = io.imread(jpg_name_path)      
        if len(jpg_img.shape)!=2:
            jpg_img = color.rgb2gray(jpg_img)
        jpg_img = transform.resize(image=jpg_img, output_shape=(out_size,out_size)) * 255
        jpg_img = np.where(nodule_label>0, jpg_img,0)
        
        jpg_h_length = max_h_list-min_h_list
        jpg_v_length = max_v_list-min_v_list
        if jpg_h_length > jpg_v_length:
            r = out_size/jpg_h_length        # Scaling ratio between nodule size and "out_size"
            jpg_img = transform.resize(image=jpg_img, output_shape=(int(nodule_label.shape[0]*r),int(nodule_label.shape[1]*r))) * 255
            for i in range(int(min_h_list*r), int(max_h_list*r)):
                for j in range(int(min_v_list*r), int(max_v_list*r)):
                    correct_length = int((out_size-jpg_v_length*r)/2)-1
                    jpg_frame_img[i-int(min_h_list*r)][j-int(min_v_list*r)+correct_length] = jpg_img[i][j]
        else:
            r = out_size/jpg_v_length
            jpg_img = transform.resize(image=jpg_img, output_shape=(int(nodule_label.shape[0]*r),int(nodule_label.shape[1]*r))) * 255
            for i in range(int(min_h_list*r), int(max_h_list*r)):
                for j in range(int(min_v_list*r), int(max_v_list*r)):
                    correct_length = int((out_size-jpg_h_length*r)/2)-1
                    jpg_frame_img[i-int(min_h_list*r)+correct_length][j-int(min_v_list*r)] = jpg_img[i][j]
        jpg_frame_img = (jpg_frame_img-np.min(jpg_frame_img))/(np.max(jpg_frame_img)-np.min(jpg_frame_img))
        jpg_frame_img = img_as_ubyte(jpg_frame_img)
    
        save_jpg_path = process_dir_path+"\\"+jpg_path
        io.imsave(arr=jpg_frame_img, fname="./temp.jpg")
        # Brightness enhancement
        img = cv2.imread("./temp.jpg")
        img = aug(img)
        cv2.imwrite(save_jpg_path, img)
    
        print("Finish {:4f}% tasks.".format(jpg_num/jpgs_num*100),end='\r')
        
def tinet_main(edge_train_bool = False, edge_cut_bool = False, texture_train_bool = False, texture_cut_bool = False,
               echo_train_bool = False, logistic_train_bool = False, test_set = "test"):
    
    config = Config()
    ####### Shape feature
    benign_dir = r".\\data\\label\\"+test_set+"\\benign"
    malignant_dir = r".\\data\\label\\"+test_set+"\\malignant"
    extractor = ShapeFeatureExtractor(dilation_value=3)
    all_ratios, num_benign, num_malignant = extractor.process_benign_and_malignant(benign_dir, malignant_dir)
    ratio = extractor.print_ratios(all_ratios, num_benign, num_malignant)

    ################### Texture Process #####################
    if texture_cut_bool:
        datasets = ['train', 'val', 'test']
    else:
        datasets = ['test']
    categories = ['malignant', 'benign']
    for dataset in datasets:
        for category in categories:
            jpg_dir_path = r".\\data\\image\\"+dataset+"\\"+category         # origin image dir
            nodule_dir_path = r".\\data\\label\\"+dataset+"\\"+category      # nodule label dir
            process_dir_path = r".\\data\\nodule\\"+dataset+"\\"+category    # nodule image dir
            process_pic(jpg_dir_path, nodule_dir_path, process_dir_path)

    ################### Edge Process #####################
    if edge_cut_bool:
        datasets = ['train', 'val', 'test']
    else:
        datasets = ['test']

    img_dir = r".\\data\\image"
    seg_dir = r".\\data\\label"
    save_dir = r".\\data\\edge_images"
    cropped_save_dir = r".\\data\\cropped_images"
    categories = ['malignant', 'benign']

    for dataset_path in datasets:
        EdgeProcess.dec_process(img_dir+"\\"+dataset_path, seg_dir+"\\"+dataset_path, save_dir+"\\"+dataset_path)
        EdgeDetectionProcess.dec_process(img_dir+"\\"+dataset_path, seg_dir+"\\"+dataset_path, cropped_save_dir+"\\"+dataset_path)

    extractor = EdgeFeatureExtractor(img_dir, seg_dir, save_dir)
    extractor.extract_and_save(datasets, categories)
    for dataset in datasets:
        for category in categories:
            img_dir_cat = os.path.join(img_dir, dataset, category)
            det_dir_cat = os.path.join(seg_dir, dataset, category)
            output_dir_cat = os.path.join(cropped_save_dir, dataset, category)
            os.makedirs(output_dir_cat, exist_ok=True)
            detection = Detection(img_dir_cat, det_dir_cat, output_dir_cat)
            detection.detect_and_crop()
    
    ####### Edge feature
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    edge_data_dir = os.path.join('.\\data', 'edge_images')

    edge_name = './weights/edge_inception_v4.pth'
    if edge_train_bool:
        edge_train_dataset = EdgeImageDataset(os.path.join(edge_data_dir, 'train'), transform=get_edge_transform())
        edge_train_dataset = EdgeImageDataset(os.path.join(edge_data_dir, 'train'), transform=get_edge_transform())
        edge_val_dataset = EdgeImageDataset(os.path.join(edge_data_dir, 'val'), transform=get_edge_transform())
        edge_dataloaders = {
            'train': DataLoader(edge_train_dataset, batch_size=128, shuffle=True, num_workers=0),
            'val': DataLoader(edge_val_dataset, batch_size=128, shuffle=True, num_workers=0),
            'test': DataLoader(edge_test_dataset, batch_size=128, shuffle=True, num_workers=0)
        }
        edge_model = get_resnet_model().to(device)
        edge_optimizer, edge_criterion, edge_num_epochs = config.texture_model_parameters(edge_model)
        edge_model_ft = train_model(edge_model, edge_name, edge_dataloaders, edge_criterion, edge_optimizer, edge_num_epochs, device)
        torch.save(edge_model_ft.state_dict(), edge_name)

    print("--Load Edge Feature Model...")
    loaded_model = load_model(edge_name)
    print("--Evaluate the test data")
    if test_set == 'train':
        edge_train_dataset = EdgeImageDataset(os.path.join(edge_data_dir, 'train'), transform=get_edge_transform())
        test_dataloader = {'test': DataLoader(edge_train_dataset, batch_size=1, shuffle=True, num_workers=0)}
        edge, label = evaluate_model(loaded_model, test_dataloader)
    elif test_set == 'test':
        edge_test_dataset = EdgeImageDataset(os.path.join(edge_data_dir, 'test'), transform=get_edge_transform())
        test_dataloader = {'test': DataLoader(edge_test_dataset, batch_size=1, shuffle=True, num_workers=0)}
        edge, label = evaluate_model(loaded_model, test_dataloader)

    ####### Echo feature
    benign_img_dir = r".\\data\\image\\"+test_set+"\\benign"
    benign_edge_dir = r".\\data\\edge_images\\"+test_set+"\\benign"
    malignant_img_dir = r".\\data\\image\\"+test_set+"\\malignant"
    malignant_edge_dir = r".\\data\\edge_images\\"+test_set+"\\malignant"
    extractor = Echogenicity(benign_img_dir, benign_edge_dir, malignant_img_dir, malignant_edge_dir, echo_train_bool)
    p_norm_echo = extractor.calculate_p_norm()

    ####### Texture feature
    texture_data_dir = os.path.join('.\\data', 'nodule')

    texture_name = './weights/texture_inception_v4.pth'
    if texture_train_bool:
        texture_train_dataset = TextureImageDataset(os.path.join(texture_data_dir, 'train'), transform=get_texture_transform()['train'])
        texture_val_dataset = TextureImageDataset(os.path.join(texture_data_dir, 'val'), transform=get_texture_transform()['val'])
        texture_test_dataset = TextureImageDataset(os.path.join(texture_data_dir, 'test'), transform=get_texture_transform()['test'])
        texture_dataloaders = {
            'train': DataLoader(texture_train_dataset, batch_size=128, shuffle=True, num_workers=0),
            'val': DataLoader(texture_val_dataset, batch_size=128, shuffle=True, num_workers=0),
            'test': DataLoader(texture_test_dataset, batch_size=128, shuffle=True, num_workers=0)
        }
        texture_model = get_resnet_model().to(device)
        texture_model.load_state_dict(torch.load(texture_name))
        texture_optimizer, texture_criterion, texture_num_epochs = config.texture_model_parameters(texture_model)
        texture_model_ft = train_model(texture_model, texture_name, texture_dataloaders, texture_criterion, texture_optimizer, texture_num_epochs, device)
        torch.save(texture_model_ft.state_dict(), texture_name)

    print("--Load Texture Feature Model...")
    loaded_model = load_model(texture_name)
    print("--Evaluate the test data")
    if test_set == 'train':
        texture_train_dataset = TextureImageDataset(os.path.join(texture_data_dir, 'train'), transform=get_texture_transform()['train'])
        test_dataloader = {'test': DataLoader(texture_train_dataset, batch_size=1, shuffle=True, num_workers=0)}
        texture, label = evaluate_model(loaded_model, test_dataloader)
    elif test_set == 'test':
        texture_test_dataset = TextureImageDataset(os.path.join(texture_data_dir, 'test'), transform=get_texture_transform()['test'])
        test_dataloader = {'test': DataLoader(texture_test_dataset, batch_size=1, shuffle=True, num_workers=0)}
        texture, label = evaluate_model(loaded_model, test_dataloader)

    # Save the feature result
    result_file = '.\\data\\data.csv'
    print()
    result_df = pd.DataFrame({
        'ratio': ratio,
        'p_norm_echo': p_norm_echo,
        'edge': edge,
        'texture': texture,
        'label': label})
    result_df.to_csv(result_file, index=False)

    # Classification
    feature_cols = ['texture', 'edge', 'p_norm_echo', 'ratio']
    label_col = ['label']
    model = LogisticRegressionWithANOVA(result_df, feature_cols, label_col, logistic_train_bool)
    model.run()


if __name__ == "__main__":
    # Parameter Explanations:
    # edge_train_bool     :Train and save the edge classification model
    # edge_cut_bool       :Process edge images and save them
    # texture_train_bool  :Train and save the internal texture classification model
    # texture_cut_bool    :Process internal texture images and save them
    # echo_train_bool     :Extract echo difference fitting function based on training data
    # logistic_train_bool :Save classification model parameters
    # For first-time use, set edge_train_bool,edge_cut_bool,texture_train_bool,texture_cut_bool,echo_train_bool,logistic_train_bool to True
    # tinet_main(edge_train_bool = True, edge_cut_bool = True, texture_train_bool = True, texture_cut_bool = True,
    #            echo_train_bool = True, logistic_train_bool = True, test_set = "train")
    # For second-time use, no need to set parameters.
    tinet_main()
