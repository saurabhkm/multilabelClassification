import torch
import utils
import torch.nn.functional as F
import numpy as np

def update(sample, labels, model, criterion, regParam, device, K, theta):
    scores, encodedVisuals, decodedVisuals, visualFeatures, encodedLabels, decodedLabels = model(sample, labels)
    # scores = model(sample, labels)
    # Top-K based label decision module
    idx_topK = scores.argsort(descending=True)
    predictedLabels = torch.zeros_like(labels)
    for i in range(scores.size(0)):
        predictedLabels[i, idx_topK[i, :K].squeeze()] = 1
    # Thesholding based decision module
    idx_thresh = torch.gt(scores, theta).type(torch.cuda.FloatTensor)
    # Combining both Top-K and Thresholding based decision modules into one.
    predictedLabels = torch.mul(predictedLabels, idx_thresh)
    # Regularizers
    # visualConsistency = criterion[0](decodedVisuals, visualFeatures)
    # labelConsistency = criterion[0](decodedLabels, labels)
    # latentAlignment = criterion[0](encodedVisuals, encodedLabels)
    classificationLoss = criterion[1](scores+10e-9, labels)
    # positiveScores = scores*labels
    # negativeScores = scores*(1-labels)
    # labelRanking = criterion[2](torch.div(torch.sum(positiveScores, dim=1), torch.sum(labels, dim=1)),
    #     torch.div(torch.sum(negativeScores, dim=1), torch.sum((1-labels), dim=1)), torch.ones(1).to(device)) # Ranking loss b/w AVG(+ve label score)+AVG(-ve label scores) - margin]
    # predictedLabels = (scores > 0.5).type(torch.cuda.FloatTensor)
    # regularizers =  labelRanking# + visualConsistency + labelConsistency + latentAlignment
    # loss = (1-regParam)*classificationLoss + regParam*regularizers
    loss = classificationLoss
    return loss, predictedLabels

def alignedBaselineUpdate(sample, labels, model, criterion, alignW, visEncW, labEncW, K, theta):
    scores, encodedVisuals, decodedVisuals, visualFeatures, encodedLabels, decodedLabels = model(sample, labels)
    # Top-K based label decision module
    idx_topK = scores.argsort(descending=True)
    predictedLabels = torch.zeros_like(labels)
    for i in range(scores.size(0)):
        predictedLabels[i, idx_topK[i, :K].squeeze()] = 1
    # Thesholding based decision module
    idx_thresh = torch.gt(scores, theta).type(torch.cuda.FloatTensor)
    # Combining both Top-K and Thresholding based decision modules into one.
    predictedLabels = torch.mul(predictedLabels, idx_thresh)
    # Regularizers
    latentAlignment = criterion[0](encodedVisuals, encodedLabels)
    classificationLoss = criterion[1](scores+10e-9, labels)
    ## Weight regularization
    # Visual encoder
    visualWeightReg = None
    for W in model.parameters():
        if visualWeightReg is None:
            visualWeightReg = W.norm(2)
        else:
            visualWeightReg = visualWeightReg + W.norm(2)
    # Label encoder
    labelWeightReg = None
    for W in model.parameters():
        if labelWeightReg is None:
            labelWeightReg = W.norm(2)
        else:
            labelWeightReg = labelWeightReg + W.norm(2)
    # Final loss term
    loss = (1 - (alignW + visEncW + labEncW))*classificationLoss + \
            alignW*latentAlignment + visEncW*visualWeightReg + labEncW*labelWeightReg
    return loss, predictedLabels

def consistentAlignedBaselineUpdate(sample, labels, model, criterion, consiW, alignW, visEncW, labEncW, K, theta):
    scores, encodedVisuals, decodedVisuals, visualFeatures, encodedLabels, decodedLabels = model(sample, labels)
    # Top-K based label decision module
    idx_topK = scores.argsort(descending=True)
    predictedLabels = torch.zeros_like(labels)
    for i in range(scores.size(0)):
        predictedLabels[i, idx_topK[i, :K].squeeze()] = 1
    # Thesholding based decision module
    idx_thresh = torch.gt(scores, theta).type(torch.cuda.FloatTensor)
    # Combining both Top-K and Thresholding based decision modules into one.
    predictedLabels = torch.mul(predictedLabels, idx_thresh)
    # Regularizers
    # visualConsistency = criterion[0](decodedVisuals, visualFeatures)
    labelConsistency = criterion[0](decodedLabels, labels)
    latentAlignment = criterion[0](encodedVisuals, encodedLabels)
    classificationLoss = criterion[1](scores+10e-9, labels)
    ## Weight regularization
    # Visual encoder
    visualWeightReg = None
    for W in model.parameters():
        if visualWeightReg is None:
            visualWeightReg = W.norm(2)
        else:
            visualWeightReg = visualWeightReg + W.norm(2)
    # Label encoder
    labelWeightReg = None
    for W in model.parameters():
        if labelWeightReg is None:
            labelWeightReg = W.norm(2)
        else:
            labelWeightReg = labelWeightReg + W.norm(2)
    # Final loss term
    loss = (1 - (consiW + alignW + visEncW + labEncW))*classificationLoss + \
            consiW*labelConsistency + alignW*latentAlignment + visEncW*visualWeightReg + labEncW*labelWeightReg
    return loss, predictedLabels

def baselineUpdate(sample, labels, model, criterion, K, theta):
    # scores, encodedVisuals, decodedVisuals, visualFeatures, encodedLabels, decodedLabels = model(sample, labels)
    scores = model(sample, labels)
    # idx_topK = scores.argsort(descending=True)
    _, idx_topK = torch.sort(scores, descending=True)
    predictedLabels = torch.zeros_like(labels)
    for i in range(scores.size(0)):
        predictedLabels[i, idx_topK[i, :K].squeeze()] = 1
    # Thesholding based decision module
    idx_thresh = torch.gt(scores, theta).type(torch.cuda.FloatTensor)
    # Combining both Top-K and Thresholding based decision modules into one.
    predictedLabels = torch.mul(predictedLabels, idx_thresh)
    # Regularizers
    classificationLoss = criterion[1](scores+10e-9, labels)
    loss = classificationLoss
    return loss, predictedLabels

def rankingUpdate(sample, labels, model, criterion, rankW, device):
    scores, encodedVisuals, decodedVisuals, visualFeatures, encodedLabels, decodedLabels = model(sample, labels)
    classificationLoss = criterion[1](scores+10e-9, labels)
    positiveScores = scores*labels
    negativeScores = scores*(1-labels)
    labelRanking = criterion[2](torch.div(torch.sum(positiveScores, dim=1), torch.sum(labels, dim=1)),
        torch.div(torch.sum(negativeScores, dim=1), torch.sum((1-labels), dim=1)), torch.ones(1).to(device)) # Ranking loss b/w AVG(+ve label score)+AVG(-ve label scores) - margin]
    predictedLabels = (scores > 0.5).type(torch.cuda.FloatTensor)
    loss = (1 - rankW)*classificationLoss + rankW*labelRanking
    return loss, predictedLabels

def alignedRankingUpdate(sample, labels, model, criterion, alignW, visEncW, labEncW, rankW, device):
    scores, encodedVisuals, encodedPositiveLabels, encodedNegativeLabels = model(sample, labels)
    # Ranking the labels
    positiveScores = scores*labels
    negativeScores = scores*(1-labels)
    labelRanking = criterion[2](torch.div(torch.sum(positiveScores, dim=1), torch.sum(labels, dim=1)),
        torch.div(torch.sum(negativeScores, dim=1), torch.sum((1-labels), dim=1)), torch.ones(1).to(device)) # Ranking loss b/w AVG(+ve label score)+AVG(-ve label scores) - margin]
    predictedLabels = (scores > 0.5).type(torch.cuda.FloatTensor)
    # Regularizers
    latentAlignment = criterion[3](encodedVisuals, encodedPositiveLabels, encodedNegativeLabels)
    embeddingSizePenalty = encodedVisuals.norm(2) + encodedPositiveLabels.norm(2) + encodedNegativeLabels.norm(2)
    classificationLoss = criterion[1](scores+10e-9, labels)
    ## Weight regularization
    # Visual encoder
    visualWeightReg = None
    for W in model.parameters():
        if visualWeightReg is None:
            visualWeightReg = W.norm(2)
        else:
            visualWeightReg = visualWeightReg + W.norm(2)
    # Label encoder
    labelWeightReg = None
    for W in model.parameters():
        if labelWeightReg is None:
            labelWeightReg = W.norm(2)
        else:
            labelWeightReg = labelWeightReg + W.norm(2)
    # Final loss term
    loss = (1 - (alignW + visEncW + labEncW + rankW))*classificationLoss + 0.001*embeddingSizePenalty + \
        alignW*latentAlignment + visEncW*visualWeightReg + labEncW*labelWeightReg + rankW*labelRanking
    return loss, predictedLabels

def alignedRankingUpdateVisualDecoder(sample, sampleX, labels, model, criterion, alignW, visEncW, labEncW, rankW, device):
    scores, encodedVisuals, decodedVisuals, crossVisuals, encodedPositiveLabels, encodedNegativeLabels = model(sample, sampleX, labels)
    # Ranking the labels
    positiveScores = scores*labels
    negativeScores = scores*(1-labels)
    labelRanking = criterion[2](torch.div(torch.sum(positiveScores, dim=1), torch.sum(labels, dim=1)),
        torch.div(torch.sum(negativeScores, dim=1), torch.sum((1-labels), dim=1)), torch.ones(1).to(device)) # Ranking loss b/w AVG(+ve label score)+AVG(-ve label scores) - margin]
    predictedLabels = (scores > 0.5).type(torch.cuda.FloatTensor)
    # Regularizers
    latentAlignment = criterion[3](encodedVisuals, encodedPositiveLabels, encodedNegativeLabels)
    embeddingSizePenalty = encodedVisuals.norm(2) + encodedPositiveLabels.norm(2) + encodedNegativeLabels.norm(2)
    classificationLoss = criterion[1](scores+10e-9, labels)
    visualConsistency = criterion[0](decodedVisuals, crossVisuals)
    ## Weight regularization
    # Visual encoder
    visualWeightReg = None
    for W in model.parameters():
        if visualWeightReg is None:
            visualWeightReg = W.norm(2)
        else:
            visualWeightReg = visualWeightReg + W.norm(2)
    # Label encoder
    labelWeightReg = None
    for W in model.parameters():
        if labelWeightReg is None:
            labelWeightReg = W.norm(2)
        else:
            labelWeightReg = labelWeightReg + W.norm(2)
    # Final loss term
    loss = (1 - (alignW + visEncW + labEncW + rankW))*classificationLoss + 0.001*embeddingSizePenalty + \
        alignW*latentAlignment + visEncW*visualWeightReg + labEncW*labelWeightReg + rankW*labelRanking + 0.05*visualConsistency
    return loss, predictedLabels

def newUpdate(sample, labels, model, criterion, alignW, visEncW, labEncW, rankW, device):
    scores, scoresMinusTheta, encodedVisuals, encodedPositiveLabels, encodedNegativeLabels = model(sample, labels)
    # Ranking the labels
    positiveScores = scores*labels
    negativeScores = scores*(1-labels)
    labelRanking = criterion[2](torch.div(torch.sum(positiveScores, dim=1), torch.sum(labels, dim=1)),
        torch.div(torch.sum(negativeScores, dim=1), torch.sum((1-labels), dim=1)), torch.ones(1).to(device)) # Ranking loss b/w AVG(+ve label score)+AVG(-ve label scores) - margin]
    # predictedLabels = (scores > 0.5).type(torch.cuda.FloatTensor)
    predictedLabels = (scoresMinusTheta > 0).type(torch.cuda.FloatTensor)
    # Regularizers
    # latentAlignment = criterion[3](encodedVisuals, torch.cat((encodedVisuals, encodedPositiveLabels), 0), torch.cat((encodedVisuals, encodedNegativeLabels), 0))
    latentAlignment = criterion[0](torch.cat((encodedVisuals, encodedVisuals), 0), torch.cat((encodedVisuals, encodedPositiveLabels), 0))
    embeddingSizePenalty = encodedVisuals.norm(2) + encodedPositiveLabels.norm(2) + encodedNegativeLabels.norm(2)
    classificationLoss = criterion[1](predictedLabels, labels)
    ## Weight regularization
    # Visual encoder
    visualWeightReg = None
    for W in model.parameters():
        if visualWeightReg is None:
            visualWeightReg = W.norm(2)
        else:
            visualWeightReg = visualWeightReg + W.norm(2)
    # Label encoder
    labelWeightReg = None
    for W in model.parameters():
        if labelWeightReg is None:
            labelWeightReg = W.norm(2)
        else:
            labelWeightReg = labelWeightReg + W.norm(2)
    # Final loss term
    loss = (1 - (alignW + visEncW + labEncW + rankW))*classificationLoss + 0.0005*embeddingSizePenalty + \
        alignW*latentAlignment + visEncW*visualWeightReg + labEncW*labelWeightReg + rankW*labelRanking
    return loss, predictedLabels

def train(trainLoader, model, criterion, optimizer, epoch, plotter, writer, logInterval, classCount, consiW=None, alignW=None, visEncW=None, labEncW=None, rankW=None, device=None, K=None, theta=None):
    totalLosses = utils.AverageMeter()
    totalTruePC = torch.zeros(classCount).to(device)
    totalPredPC = torch.zeros(classCount).to(device)
    totalCorrectPC = torch.zeros(classCount).to(device)
    exactMatches = 0

    model.train()   # switch to train mode
    # for batchIdx, (sample, sampleX, labels) in enumerate(trainLoader):
    for batchIdx, (sample, labels) in enumerate(trainLoader):
        # sample, sampleX, labels = sample.to(device), sampleX.to(device), labels.to(device)
        sample, labels = sample.to(device), labels.to(device)
        # Update function
        loss, predictedLabels = baselineUpdate(sample, labels, model, criterion, K, theta)
        # loss, predictedLabels = consistentAlignedBaselineUpdate(sample, labels, model, criterion, consiW, alignW, visEncW, labEncW, K, theta)
        # loss, predictedLabels = alignedBaselineUpdate(sample, labels, model, criterion, alignW, visEncW, labEncW, K, theta)
        # loss, predictedLabels = rankingUpdate(sample, labels, model, criterion, rankW, device)
        # loss, predictedLabels = alignedRankingUpdate(sample, labels, model, criterion, alignW, visEncW, labEncW, rankW, device)
        # loss, predictedLabels = alignedRankingUpdateVisualDecoder(sample, sampleX, labels, model, criterion, alignW, visEncW, labEncW, rankW, device)
        # loss, predictedLabels = newUpdate(sample, labels, model, criterion, alignW, visEncW, labEncW, rankW, device)
        ## Record performance
        totalLosses.update(loss.item(), sample.size(0))
        totalCorrectPC, totalTruePC, totalPredPC, exactMatches = utils.batchPerformance(predictedLabels, labels, totalCorrectPC, totalTruePC, totalPredPC, exactMatches)
        
        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batchIdx % logInterval == 0:
            print('Train- Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'.format(
                epoch, batchIdx * len(sample), len(trainLoader.dataset),
                totalLosses.val, totalLosses.avg))
    pcPrecision, pcRecall, pcF1, ovPrecision, ovRecall, ovF1, zeroOneScore = utils.measurePerformance(totalCorrectPC, totalPredPC, totalTruePC, classCount, exactMatches, len(trainLoader.dataset))

    # log avg values to somewhere
    if writer != None:
        writer.writePerformance(['train', epoch, totalLosses.avg, pcPrecision.item(), pcRecall.item(),
                                ovPrecision, ovRecall, zeroOneScore, pcF1.item(), ovF1])
    # plot avg values to somewhere
    if plotter != None:
        plotter.plot('loss', 'train', epoch, totalLosses.avg)
        plotter.plot('Per-class Precision', 'train', epoch, pcPrecision.item())
        plotter.plot('Per-class Recall', 'train', epoch, pcRecall.item())
        plotter.plot('Overall Precision', 'train', epoch, ovPrecision)
        plotter.plot('Overall Recall', 'train', epoch, ovRecall)
        plotter.plot('0/1 Score', 'train', epoch, zeroOneScore)
        plotter.plot('Per-class F1', 'train', epoch, pcF1.item())
        plotter.plot('Overall F1', 'train', epoch, ovF1)

def test(testLoader, model, criterion, epoch, plotter, writer, classCount, consiW=None, alignW=None, visEncW=None, labEncW=None, rankW=None, device=None, K=None, theta=None):
    totalLosses = utils.AverageMeter()
    totalTruePC = torch.zeros(classCount).to(device)
    totalPredPC = torch.zeros(classCount).to(device)
    totalCorrectPC = torch.zeros(classCount).to(device)
    exactMatches = 0

    model.eval()    # switch to evaluation mode
    # for batch_idx, (sample, sampleX, labels) in enumerate(testLoader):
    for batchIdx, (sample, labels) in enumerate(testLoader):
        # sample, sampleX, labels = sample.to(device), sampleX.to(device), labels.to(device)
        sample, labels = sample.to(device), labels.to(device)
        # Update function
        loss, predictedLabels = baselineUpdate(sample, labels, model, criterion, K, theta)
        # loss, predictedLabels = consistentAlignedBaselineUpdate(sample, labels, model, criterion, consiW, alignW, visEncW, labEncW, K, theta)
        # loss, predictedLabels = alignedBaselineUpdate(sample, labels, model, criterion, alignW, visEncW, labEncW, K, theta)
        # loss, predictedLabels = rankingUpdate(sample, labels, model, criterion, rankW, device)
        # loss, predictedLabels = alignedRankingUpdate(sample, labels, model, criterion, alignW, visEncW, labEncW, rankW, device)
        # loss, predictedLabels = alignedRankingUpdateVisualDecoder(sample, sampleX, labels, model, criterion, alignW, visEncW, labEncW, rankW, device)
        # loss, predictedLabels = newUpdate(sample, labels, model, criterion, alignW, visEncW, labEncW, rankW, device)
        ## Record Performance
        totalLosses.update(loss.item(), sample.size(0))
        totalCorrectPC, totalTruePC, totalPredPC, exactMatches = utils.batchPerformance(predictedLabels, labels, totalCorrectPC, totalTruePC, totalPredPC, exactMatches)

    pcPrecision, pcRecall, pcF1, ovPrecision, ovRecall, ovF1, zeroOneScore = utils.measurePerformance(totalCorrectPC, totalPredPC, totalTruePC, classCount, exactMatches, len(testLoader.dataset))

    print('Test- Loss: {:.4f}, pcP: {:.1f}, pcR: {:.1f}, ovP: {:.1f}, ovR: {:.1f}, pcF1: {:.1f}, 0-1: {:.1f}\n '.format(
        totalLosses.avg, 100*pcPrecision, 100*pcRecall, 100*ovPrecision, 100*ovRecall, 100*pcF1, 100*zeroOneScore))
    if writer != None:
        writer.writePerformance(['test', epoch, totalLosses.avg, pcPrecision.item(), pcRecall.item(),
                                ovPrecision, ovRecall, zeroOneScore, pcF1.item(), ovF1])
    if plotter != None:
        plotter.plot('loss', 'test', epoch, totalLosses.avg)
        plotter.plot('Per-class Precision', 'test', epoch, pcPrecision.item())
        plotter.plot('Per-class Recall', 'test', epoch, pcRecall.item())
        plotter.plot('Overall Precision', 'test', epoch, ovPrecision)
        plotter.plot('Overall Recall', 'test', epoch, ovRecall)
        plotter.plot('0/1 Score', 'test', epoch, zeroOneScore)
        plotter.plot('Per-class F1', 'test', epoch, pcF1.item())
        plotter.plot('Overall F1', 'test', epoch, ovF1)
    return zeroOneScore