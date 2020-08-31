import torch, os, collections, random, sys
import numpy as np
import pandas as pd
from torchvision import transforms
from PIL import Image, ImageFile
from torch.utils.data import Dataset
ImageFile.LOAD_TRUNCATED_IMAGES = True
if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

#### Raw datasets ------------------------------------
class MSCOCO(Dataset):
    vip_root = '/home/SharedData/saurabh/MLC/MSCOCO'
    spacetime_root = '/home/viplab/saurabh/data/MLC/MSCOCO'
    trainImageDir = 'train2014'
    trainLabelsFile = 'annotations/instances_train2014.json'
    valImageDir = 'val2014'
    valLabelsFile = 'annotations/instances_val2014.json'

    def __init__(self, train=True, transform=None, spacetime=False):
        from pycocotools.coco import COCO
        self.root = self.spacetime_root if spacetime else self.vip_root
        if train == True:
            self.imgDir = os.path.join(self.root, self.trainImageDir)
            self.annFile = os.path.join(self.root, self.trainLabelsFile)
        elif train == False:
            self.imgDir = os.path.join(self.root, self.valImageDir)
            self.annFile = os.path.join(self.root, self.valLabelsFile)
        self.coco = COCO(self.annFile)
        self.correctLabels = self.coco.getCatIds()
        self.ids = list(self.coco.imgs.keys())
        self.validIDs = self._getValidIDs(self.ids)
        self.transform = transform

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.validIDs[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]['file_name']

        img = Image.open(os.path.join(self.imgDir, path)).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        multilabel = self._labelList2multilabel(self._target2labels(target))

        return img, multilabel

    def __len__(self):
        return len(self.validIDs)

    def _target2labels(self, target):
        labelList = []
        for i in range(len(target)):
            label = target[i]['category_id']
            correctedLabel = self.correctLabels.index(label)
            labelList.append(correctedLabel)
        return labelList

    def _labelList2multilabel(self, labelList):
        labelList = np.unique(labelList)
        labels = torch.LongTensor(labelList).unsqueeze_(1)
        multilabel = torch.zeros(len(labelList), 80)
        multilabel.scatter_(1, labels, 1)
        multilabel = multilabel.sum(dim=0)
        return multilabel

    def _getValidIDs(self, ids):
        validIDs = []
        for index in range(len(ids)):
            img_id = ids[index]
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = self.coco.loadAnns(ann_ids)
            labels = self._target2labels(target)
            if len(labels)!=0:
                validIDs.append(img_id)
        return validIDs


def getMSCOCOLoaders(train_batch_size, test_batch_size, spacetime, **kwargs):
    classCount = 80
    trainTransform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testTransform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainDataset = MSCOCO(train=True, transform=trainTransform, spacetime=spacetime)
    testDataset = MSCOCO(train=False, transform=testTransform, spacetime=spacetime)

    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=test_batch_size, shuffle=False, **kwargs)

    return trainLoader, testLoader, classCount



class PASCALVOC(Dataset):
    """`Pascal VOC <http://host.robots.ox.ac.uk/pascal/VOC/>`_ Detection Dataset.
    Args:
        root (string): Root directory of the VOC Dataset.
        year (string, optional): The dataset year, supports years 2007 to 2012.
        image_set (string, optional): Select the image_set to use, ``train``, ``trainval`` or ``val``
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, required): A function/transform that takes in the
            target and transforms it.
    """
    vip_root = '/home/SharedData/saurabh/MLC/PASCALVOC/'
    spacetime_root = '/home/viplab/saurabhkm/data/MLC/PASCALVOC/'
    classNames =['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, train=True, transform=None, spacetime=False):
        self.root = self.spacetime_root if spacetime else self.vip_root
        self.transform = transform
        if train == True:
            self.image_set='trainval'
        else:
            self.image_set='test'

        base_dir = 'VOCdevkit/VOC2007'
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')
        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.transform is not None:
            img = self.transform(img)

        multilabel = self._labelList2multilabel(self._target2labels(target))

        return img, multilabel


    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                {ind: v[0] if len(v) == 1 else v
                 for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def _target2labels(self, target):
        labels = []
        metalist = target['annotation']['object']
        if isinstance(metalist, dict):
            label = metalist['name']
            labels.append(self.classNames.index(label))
        if isinstance(metalist, list):
            for i in range(len(metalist)):
                label = metalist[i]['name']
                labels.append(self.classNames.index(label))
        return labels

    def _labelList2multilabel(self, labelList):
        labelList = np.unique(labelList)
        labels = torch.LongTensor(labelList).unsqueeze_(1)
        multilabel = torch.zeros(len(labelList), 20)
        multilabel.scatter_(1, labels, 1)
        multilabel = multilabel.sum(dim=0)
        return multilabel


def getPASCALLoaders(train_batch_size, test_batch_size, spacetime, **kwargs):
    classCount = 20
    trainTransform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testTransform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainDataset = PASCALVOC(train=True, transform=trainTransform, spacetime=spacetime)
    testDataset = PASCALVOC(train=False, transform=testTransform, spacetime=spacetime)

    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=test_batch_size, shuffle=False, **kwargs)

    return trainLoader, testLoader, classCount

class NUSWIDE(Dataset):
    vip_root = '/home/SharedData/saurabh/MLC/NUSWIDE/'
    spacetime_root = '/home/viplab/saurabh/data/MLC/NUSWIDE/'

    def __init__(self, train=True, transform=None, spacetime=False):
        self.transform = transform
        self.train = train
        self.root = self.spacetime_root if spacetime else self.vip_root
        
        # Load pickled dataframe
        dataset = pd.read_pickle('../makeDatasets/NUSWIDE/nuswideDataframe.pkl')
        testDF = dataset.sample(n=50261, random_state=42)
        trainDF = dataset.drop(testDF.index)
        if train == True:
            self.DF = trainDF
            self.indices = trainDF.index
        elif train == False:
            self.DF = testDF
            self.indices = testDF.index
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        image = Image.open(self.root + 'Flickr/' + self.DF.loc[index][0].replace('\\', '/')).convert('RGB')
        multilabel = torch.FloatTensor(self.DF.loc[index][1:-4])
        if self.transform:
            image = self.transform(image)
        
        return image, multilabel

    def __len__(self):
        return len(self.indices)


def getNUSWIDELoaders(train_batch_size, test_batch_size, spacetime, **kwargs):
    classCount = 81
    trainTransform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testTransform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainDataset = NUSWIDE(train=True, transform=trainTransform, spacetime=spacetime)
    testDataset = NUSWIDE(train=False, transform=testTransform, spacetime=spacetime)

    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=test_batch_size, shuffle=False, **kwargs)

    return trainLoader, testLoader, classCount


#### Cross Visual Datasets ---------------------------------------
class PASCALVOC_CrossVisual(Dataset):
    vip_root = '/home/SharedData/saurabh/MLC/PASCALVOC/'
    spacetime_root = '/home/viplab/saurabhkm/data/MLC/PASCALVOC/'
    classNames =['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    def __init__(self, train=True, transform=None, spacetime=False):
        self.root = self.spacetime_root if spacetime else self.vip_root
        self.transform = transform
        if train == True:
            self.image_set='trainval'
        else:
            self.image_set='test'

        base_dir = 'VOCdevkit/VOC2007'
        voc_root = os.path.join(self.root, base_dir)
        image_dir = os.path.join(voc_root, 'JPEGImages')
        annotation_dir = os.path.join(voc_root, 'Annotations')
        splits_dir = os.path.join(voc_root, 'ImageSets/Main')
        split_f = os.path.join(splits_dir, self.image_set.rstrip('\n') + '.txt')

        with open(os.path.join(split_f), "r") as f:
            file_names = [x.strip() for x in f.readlines()]

        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.annotations = [os.path.join(annotation_dir, x + ".xml") for x in file_names]
        assert (len(self.images) == len(self.annotations))

    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = self.parse_voc_xml(
            ET.parse(self.annotations[index]).getroot())

        if self.transform is not None:
            sample = self.transform(img)
            # sampleX = self.transform(img)

        multilabel = self._labelList2multilabel(self._target2labels(target))

        # return sample, sampleX, multilabel
        return sample, multilabel


    def __len__(self):
        return len(self.images)

    def parse_voc_xml(self, node):
        voc_dict = {}
        children = list(node)
        if children:
            def_dic = collections.defaultdict(list)
            for dc in map(self.parse_voc_xml, children):
                for ind, v in dc.items():
                    def_dic[ind].append(v)
            voc_dict = {
                node.tag:
                {ind: v[0] if len(v) == 1 else v
                 for ind, v in def_dic.items()}
            }
        if node.text:
            text = node.text.strip()
            if not children:
                voc_dict[node.tag] = text
        return voc_dict

    def _target2labels(self, target):
        labels = []
        metalist = target['annotation']['object']
        if isinstance(metalist, dict):
            label = metalist['name']
            labels.append(self.classNames.index(label))
        if isinstance(metalist, list):
            for i in range(len(metalist)):
                label = metalist[i]['name']
                labels.append(self.classNames.index(label))
        return labels

    def _labelList2multilabel(self, labelList):
        labelList = np.unique(labelList)
        labels = torch.LongTensor(labelList).unsqueeze_(1)
        multilabel = torch.zeros(len(labelList), 20)
        multilabel.scatter_(1, labels, 1)
        multilabel = multilabel.sum(dim=0)
        return multilabel

def getPASCAL_CV_Loaders(train_batch_size, test_batch_size, spacetime, **kwargs):
    classCount = 20
    trainTransform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    testTransform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    trainDataset = PASCALVOC_CrossVisual(train=True, transform=trainTransform, spacetime=spacetime)
    testDataset = PASCALVOC_CrossVisual(train=False, transform=testTransform, spacetime=spacetime)

    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=test_batch_size, shuffle=False, **kwargs)

    return trainLoader, testLoader, classCount

#### NUMPY dumps --------------------------------------
class genericDataset(Dataset):
    pascalPath = '/home/SharedData/saurabh/MLC/PASCALVOC/numpyDumps/'
    mscocoPath = '/home/SharedData/saurabh/MLC/MSCOCO/numpyDumps/'
    nuswidePath = '/home/SharedData/saurabh/MLC/NUSWIDE/numpyDumps/'
    def __init__(self, name, train=True):
        if name == 'pascal':
            if train==True:
                self.data = np.load(self.pascalPath+'pascalTrainFeatures.npy')
                self.labels = np.load(self.pascalPath+'pascalTrainLabels.npy')
            elif train==False:
                self.data = np.load(self.pascalPath+'pascalTestFeatures.npy')
                self.labels = np.load(self.pascalPath+'pascalTestLabels.npy')
        if name == 'mscoco':
            if train==True:
                self.data = np.load(self.mscocoPath+'mscocoTrainFeatures.npy')
                self.labels = np.load(self.mscocoPath+'mscocoTrainLabels.npy')
            elif train==False:
                self.data = np.load(self.mscocoPath+'mscocoTestFeatures.npy')
                self.labels = np.load(self.mscocoPath+'mscocoTestLabels.npy')
        if name == 'nuswide':
            if train==True:
                self.data = np.load(self.nuswidePath+'nuswideTrainFeatures.npy')
                self.labels = np.load(self.nuswidePath+'nuswideTrainLabels.npy')
            elif train==False:
                self.data = np.load(self.nuswidePath+'nuswideTestFeatures.npy')
                self.labels = np.load(self.nuswidePath+'nuswideTestLabels.npy')

    def __getitem__(self, idx):
        sample = torch.Tensor(self.data[idx])
        label = torch.FloatTensor(self.labels[idx])
        return sample, label

    def __len__(self):
        return self.data.shape[0]

def getGenericLoaders(name, train_batch_size, test_batch_size, **kwargs):
    if name=='pascal':
        classCount = 20
    elif name=='mscoco':
        classCount = 80
    elif name=='nuswide':
        classCount = 81

    trainDataset = genericDataset(name = name, train=True)
    testDataset = genericDataset(name = name, train=False)

    trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=train_batch_size, shuffle=True, **kwargs)
    testLoader = torch.utils.data.DataLoader(testDataset, batch_size=test_batch_size, shuffle=False, **kwargs)

    return trainLoader, testLoader, classCount