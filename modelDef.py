import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

#### Label prediction modules
class labelPrediction_Cross(nn.Module):
    def __init__(self, featureExtractor, visualAutoEncoder, labelAutoEncoder, classifier):
        super(labelPrediction, self).__init__()
        self.featureExtractor = featureExtractor
        self.visualAutoEncoder = visualAutoEncoder
        self.labelAutoEncoder = labelAutoEncoder
        self.classifier = classifier

    def getNegativeLabel(self, label):
        invertedLabels = (1 - label)
        negLabels = torch.empty_like(invertedLabels)
        for batch in range(invertedLabels.size(0)):
            invLabel = invertedLabels[batch]
            invLabelIdx = invLabel.nonzero().squeeze()
            randCount = np.random.choice(np.arange(1,5))
            negLabelIdx = np.random.choice(invLabelIdx.cpu().numpy(), randCount, replace=False)
            negLabel = torch.zeros_like(invLabel)
            negLabel[negLabelIdx] = 1
            negLabels[batch] = negLabel
        return negLabels

    def forward(self, visual, visualX, label):
        visualFeatures = self.featureExtractor(visual)
        crossVisualsFeatures = self.featureExtractor(visualX)
        encodedVisuals, decodedVisuals = self.visualAutoEncoder(visualFeatures)
        encodedPositiveLabels = self.labelAutoEncoder(label)
        negLabel = self.getNegativeLabel(label)
        encodedNegativeLabels = self.labelAutoEncoder(negLabel)
        scores = self.classifier(visualFeatures)
        return scores, encodedVisuals, decodedVisuals, crossVisualsFeatures, encodedPositiveLabels, encodedNegativeLabels

class labelPrediction_cross2(nn.Module):
    def __init__(self, featureExtractor, visualEncoder, labelEncoder, classifier, thresholdEstimator):
        super(labelPrediction, self).__init__()
        self.featureExtractor = featureExtractor
        self.visualEncoder = visualEncoder
        self.labelEncoder = labelEncoder
        self.classifier = classifier
        self.thresholdEstimator = thresholdEstimator

    def getNegativeLabel(self, label):
        invertedLabels = (1 - label)
        negLabels = torch.empty_like(invertedLabels)
        for batch in range(invertedLabels.size(0)):
            invLabel = invertedLabels[batch]
            invLabelIdx = invLabel.nonzero().squeeze()
            randCount = np.random.choice(np.arange(1,5))
            negLabelIdx = np.random.choice(invLabelIdx.cpu().numpy(), randCount, replace=False)
            negLabel = torch.zeros_like(invLabel)
            negLabel[negLabelIdx] = 1
            negLabels[batch] = negLabel
        return negLabels

    def forward(self, visual, label):
        visualFeatures = self.featureExtractor(visual)
        encodedVisuals = self.visualEncoder(visualFeatures)
        encodedPositiveLabels = self.labelEncoder(label)
        negLabel = self.getNegativeLabel(label)
        encodedNegativeLabels = self.labelEncoder(negLabel)
        scores = self.classifier(encodedVisuals)
        thresholds = self.thresholdEstimator(encodedVisuals)
        return scores, scores.sub(thresholds), encodedVisuals, encodedPositiveLabels, encodedNegativeLabels
        # return scores, encodedVisuals, encodedPositiveLabels, encodedNegativeLabels

class labelPrediction(nn.Module):
    def __init__(self, featureExtractor, visualEncoder, labelEncoder, classifier):
    # def __init__(self, visualEncoder, labelEncoder, classifier):
        super(labelPrediction, self).__init__()
        self.featureExtractor = featureExtractor
        self.visualEncoder = visualEncoder
        self.labelEncoder = labelEncoder
        self.classifier = classifier

    def forward(self, visual, label):
    # def forward(self, visualFeatures, label):
        visualFeatures = self.featureExtractor(visual)
        encodedVisuals = self.visualEncoder(visualFeatures)
        encodedLabels = self.labelEncoder(label)
        scores = self.classifier(encodedVisuals)
        return scores

class visualAutoEncoder(nn.Module):
    def __init__(self):
        super(visualAutoEncoder, self).__init__()
        self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 512)
        self.fc6 = nn.Linear(512, 1024)
        self.fc7 = nn.Linear(1024, 2048)
        self.fc8 = nn.Linear(2048, 4096)
        self._initializeWeights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        y = F.relu(self.fc5(x))
        y = F.relu(self.fc6(y))
        y = F.relu(self.fc7(y))
        y = self.fc8(y)
        return x, torch.sigmoid(y)

    def _initializeWeights(self):
        # This initializes weights for all the submodules(bit & snap encoders/decoders)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

class visualEncoder(nn.Module):
    def __init__(self):
        super(visualEncoder, self).__init__()
        # self.fc1 = nn.Linear(4096, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.fc4 = nn.Linear(512, 256)
        self._initializeWeights()

    def forward(self, x):
        # x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def _initializeWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

class labelAutoEncoder(nn.Module):
    def __init__(self, numClasses):
        super(labelAutoEncoder, self).__init__()
        self.fc1 = nn.Linear(numClasses, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, numClasses)
        self._initializeWeights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = F.relu(self.fc3(x))
        y = self.fc4(y)
        return x, torch.sigmoid(y)

    def _initializeWeights(self):
        # This initializes weights for all the submodules(bit & snap encoders/decoders)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)

class labelEncoder(nn.Module):
    def __init__(self, numClasses):
        super(labelEncoder, self).__init__()
        self.fc1 = nn.Linear(numClasses, 128)
        self.fc2 = nn.Linear(128, 256)
        self._initializeWeights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def _initializeWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

class classifier(nn.Module):
    def __init__(self, numClasses):
        super(classifier, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, numClasses)
        self._initializeWeights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

    def _initializeWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

#### Visual feature extraction modules
# VGG16 features extraction module   (Upgrade to ResNet once all is working!!, Change normalization if so)
class vggFeatureExtractor(nn.Module):
    def __init__(self, extractorName):
        super(vggFeatureExtractor, self).__init__()
        if extractorName=='vgg11':
            model = torchvision.models.vgg11(pretrained=True)
        elif extractorName=='vgg13':
            model = torchvision.models.vgg13(pretrained=True)
        elif extractorName=='vgg16':
            model = torchvision.models.vgg16(pretrained=True)
        elif extractorName=='vgg19':
            model = torchvision.models.vgg19(pretrained=True)
        self.features = torch.nn.Sequential(*list(model.features.children()))
        self.classifier = torch.nn.Sequential(*list(model.classifier.children())[:-3])
        # Freeze wieghts to avoid training
        for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        out = self.classifier(x)
        return out

class resnetFeatureExtractor(nn.Module):
    def __init__(self, extractorName):
        super(resnetFeatureExtractor, self).__init__()
        if extractorName=='resnet18':
            model = torchvision.models.resnet18(pretrained=True)
        elif extractorName=='resnet34':
            model = torchvision.models.resnet34(pretrained=True)
        elif extractorName=='resnet50':
            model = torchvision.models.resnet50(pretrained=True)
        elif extractorName=='resnet101':
            model = torchvision.models.resnet101(pretrained=True)
        elif extractorName=='resnet152':
            model = torchvision.models.resnet152(pretrained=True)
        self.features = torch.nn.Sequential(*list(model.children())[:-1])
        # Freeze wieghts to avoid training
        for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        out = x.view(x.size(0), -1)
        return out


#### Label Decision
# Threshold Prediction
class thresholdEstimator(nn.Module):
    def __init__(self, numClasses):
        super(thresholdEstimator, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, numClasses)
        self._initializeWeights()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)

    def _initializeWeights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)