import torch
from dinov2.hub.backbones import dinov2_vits14
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from sklearn import svm
import joblib
from sklearn.metrics import classification_report

clsnames = ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'AGC']

def gene_npz():
    root_dir = 'data_resource/cls_pn/cut_img'

    device = torch.device('cuda:0')
    backbone_model: torch.nn.Module = dinov2_vits14(pretrained=False)
    state_dict = torch.load('checkpoints/dinov2_vits14_pretrain.pth')
    print(backbone_model.load_state_dict(state_dict, strict=True))
    backbone_model.eval()
    backbone_model.cuda()

    transform_image = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])

    for mode in ['train', 'val']:
        with open(f'{root_dir}/{mode}_rcp_c6.txt', 'r') as txtf:
            datalines = txtf.readlines()
        embeddings, filepath, labels = [],[],[]
        with torch.no_grad():
            for line in tqdm(datalines):
                clsname = line.split('/')[0]
                filename = line.split('/')[1].split(' ')[0]
                clsid = int(line.split(' ')[1].strip())
                labels.append(clsid)
                img_path = f'{root_dir}/random_cut/{clsname}/{filename}'
                img = Image.open(img_path)
                transformed_img = transform_image(img)[:3].unsqueeze(0)
                embed = backbone_model(transformed_img.to(device))
                embeddings.append(np.array(embed[0].cpu().numpy()).reshape(1, -1).tolist())
            filepath.append(img_path)
        
        save_path = f'data_resource/{mode}_rcp_c6.npz'
        np.savez(save_path, embeddings=embeddings, 
             labels=np.array(labels), filepath=np.array(labels, dtype=object))

def svm_classify():
    clf = svm.SVC(gamma='scale')
    train_data = np.load(f'data_resource/train_rcp_c6.npz', allow_pickle=True)
    val_data = np.load(f'data_resource/val_rcp_c6.npz', allow_pickle=True)

    y = train_data['labels'].tolist()
    embedding_list = train_data['embeddings']
    clf.fit(np.array(embedding_list).reshape(-1, 384), y)

    joblib.dump(clf, 'checkpoints/svm_classifier.pkl')

    preds = clf.predict(val_data['embeddings'].reshape(-1, 384))
    y_true = val_data['labels'].tolist()
    print(classification_report(y_true, preds))



if __name__ == '__main__':
    # gene_npz()
    svm_classify()