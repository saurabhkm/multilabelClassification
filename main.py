import torch, argparse
import modelDef, dataDef, losses, engine, utils

# Training settings
parser = argparse.ArgumentParser(description='Consistency driven Multi-Lable Image Classification')
parser.add_argument('-type', default='', type=str, help='Dataset name')
parser.add_argument('-batch-size', type=int, default=2, help='input batch size for training')
parser.add_argument('-test-batch-size', type=int, default=2, help='input batch size for testing')
parser.add_argument('-epochs', type=int, default=5, help='number of epochs to train')
parser.add_argument('-lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('-log', type=int, default=200, help='#batches to wait before logging training status')
parser.add_argument('-margin', type=float, default=None, help='margin for triplet loss')
parser.add_argument('-resume', default='', type=str, help='path to latest checkpoint')
parser.add_argument('-name', default='default', type=str, help='name of experiment')
parser.add_argument('-plot', action="store_true", default=False, help='enable plotting')
parser.add_argument('-spacetime', action="store_true", default=False, help='Running on spacetime')
parser.add_argument('-write', action="store_true", default=False, help='enable logging to file')
parser.add_argument('-consiW', type=float, default=None, help='Consistency penalty')
parser.add_argument('-alignW', type=float, default=None, help='Latent alignment penalty')
parser.add_argument('-visEncW', type=float, default=None, help='Visual encoder weight L2 penalty')
parser.add_argument('-labEncW', type=float, default=None, help='Label encoder weight L2 penalty')
parser.add_argument('-rankW', type=float, default=None, help='Ranking penalty')
parser.add_argument('-topK', type=int, default=None, help='K value for top-K label decision')
parser.add_argument('-theta', type=float, default=None, help='Theta value for label decision threshold')

args = parser.parse_args()
device = torch.device('cuda')

torch.manual_seed(1)
torch.cuda.manual_seed(1)
torch.backends.cudnn.benchmark = True

plotter = None
if args.plot == True:
    plotter = utils.VisdomPlotter(env_name=args.name)
    plotter.argsTile(args.__dict__)

writer = None
if args.write == True:
    writer = utils.diskWriter(directory=args.name)
    writer.writeParameters(args.__dict__)
    writer.writePerformance(['trainFlag', 'epoch', 'loss', 'pcPrecision', 'pcRecall',
                                'ovPrecision', 'ovRecall', 'zeroOneScore', 'pcF1', 'ovF1'])

kwargs = {'num_workers':8, 'pin_memory':True}

## Dataloaders
if args.type == 'pascal':
    trainLoader, testLoader, classCount = dataDef.getPASCALLoaders(args.batch_size, args.test_batch_size, args.spacetime, **kwargs)
    # trainLoader, testLoader, classCount = dataDef.getGenericLoaders(args.type, args.batch_size, args.test_batch_size, **kwargs)
elif args.type == 'mscoco':
    trainLoader, testLoader, classCount = dataDef.getMSCOCOLoaders(args.batch_size, args.test_batch_size, args.spacetime, **kwargs)
elif args.type == 'nuswide':
    trainLoader, testLoader, classCount = dataDef.getNUSWIDELoaders(args.batch_size, args.test_batch_size, args.spacetime, **kwargs)

## Models
# featureExtractor = modelDef.vggFeatureExtractor('vgg16')
featureExtractor = modelDef.resnetFeatureExtractor('resnet101')
# visualAutoEncoder = modelDef.visualAutoEncoder()
visualAutoEncoder = modelDef.visualEncoder()
# labelAutoEncoder = modelDef.labelAutoEncoder(classCount)
labelAutoEncoder = modelDef.labelEncoder(classCount)
classifier = modelDef.classifier(classCount)
# thresholdEstimator = modelDef.thresholdEstimator(classCount)
model = modelDef.labelPrediction(featureExtractor, visualAutoEncoder, labelAutoEncoder, classifier)#, thresholdEstimator)
model.to(device)

# optionally resume from a checkpoint
startEpoch = 1
if args.resume:
    startEpoch, best_prec1, stateDict = utils.load_checkpoint(args.resume)
    model.load_state_dict(stateDict)

## Losses
tripletLoss = torch.nn.TripletMarginLoss(margin = args.margin)
mseLoss = torch.nn.MSELoss()
bceLoss = torch.nn.BCELoss()
ceLoss = torch.nn.CrossEntropyLoss()
# mlmLoss = torch.nn.MultiLabelMarginLoss()
# mlsmLoss = torch.nn.MultiLabelSoftMarginLoss()
# l1Loss = torch.nn.L1Loss()
kldLoss = torch.nn.KLDivLoss()
warpLoss = losses.WARPLoss()
lsepLoss = losses.LSEPLoss()
rankingLoss = torch.nn.MarginRankingLoss(margin=args.margin)
criterion = [mseLoss, bceLoss, rankingLoss, tripletLoss, warpLoss, lsepLoss, ceLoss]

# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, amsgrad=True)

bestAcc = 0
for epoch in range(startEpoch, args.epochs + 1):
    engine.train(trainLoader, model, criterion, optimizer, epoch, plotter, writer, args.log, classCount, args.consiW, args.alignW, args.visEncW, args.labEncW, args.rankW, device, args.topK, args.theta)
    acc = engine.test(testLoader, model, criterion, epoch, plotter, writer, classCount, args.consiW, args.alignW, args.visEncW, args.labEncW, args.rankW, device, args.topK, args.theta)

    # remember best acc and save checkpoint
    is_best = acc > bestAcc
    bestAcc = max(acc, bestAcc)
    utils.save_checkpoint({'epoch': epoch + 1,
        'stateDict': model.state_dict(), 'perfMetric': bestAcc, }, is_best, args.name)