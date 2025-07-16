from itertools import cycle
import cv2 as cv
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import roc_curve, confusion_matrix
from sklearn import metrics


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plot_results():
    eval1 = np.load('Eval_all_KFold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Terms = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Classifier = ['TERMS', 'CNN', 'RAN', 'MANet', 'GRU', 'PROPOSED']
    for i in range(eval1.shape[0]):
        for m in range(eval1.shape[1]):
            value1 = eval1[i, m, :, 4:]

            Table = PrettyTable()
            Table.add_column(Classifier[0], Terms[:3])
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value1[j, :3])
            print('-------------------------------------------------- Dataset -', str(i + 1), 'Fold - ', str(m + 1),
                  ' -  Method Comparison',
                  '--------------------------------------------------')
            print(Table)


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'RDA', 'SGO', 'TOA', 'HSOA', 'PROPOSED']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Dataset = ['HAM10000', 'PH2Dataset']
    for i in range(2):
        Conv_Graph = np.zeros((5, 5))
        for j in range(5):  # for 5 algms
            Conv_Graph[j, :] = Statistical(Fitness[i, j, :])

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report for ', Dataset[i],
              '--------------------------------------------------')
        print(Table)

        length = np.arange(50)
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
                 markersize=12, label='RDA-OMUNet++')
        plt.plot(length, Conv_Graph[1, :], color=[0, 0.5, 0.5], linewidth=3, marker='*', markerfacecolor='green',
                 markersize=12, label='SGO-OMUNet++')
        plt.plot(length, Conv_Graph[2, :], color=[0.5, 0, 0.5], linewidth=3, marker='*', markerfacecolor='blue',
                 markersize=12, label='TOA-OMUNet++')
        plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
                 markersize=12, label='HSO-OMUNet++')
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
                 markersize=12, label='EHSO-OMUNet++')
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Conv_%s.png" % (i + 1))
        plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['CNN', 'RAN', 'MANet', 'GRU', 'MANet-GRU']
    for a in range(2):  # For 2 Datasets
        Actual = np.load('Target_' + str(a + 1) + '.npy', allow_pickle=True).astype('int')
        colors = cycle(["blue", "darkorange", "cornflowerblue", "deeppink", "black"])  # "aqua",
        for i, color in zip(range(5), colors):  # For all classifiers
            Predicted = np.load('Y_Score_' + str(a + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i], )

        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title('Accuracy')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path1 = "./Results/Dataset_%s_ROC.png" % (a + 1)
        plt.savefig(path1)
        plt.show()


def plot_Segmentation_results_1():
    for n in range(2):
        Eval_all = np.load('Eval_all_Segmentation_' + str(n + 1) + '.npy', allow_pickle=True)
        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD', 'VARIANCE']
        Algorithm = ['TERMS', 'RDA', 'SGO', 'TOA', 'HSOA', 'PROPOSED']
        Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR',
                 'NPV',
                 'FDR', 'F1-Score', 'MCC']

        # for n in range(Eval_all.shape[0]):
        value_all = Eval_all

        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 4))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    # stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(stats.shape[2])

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.82, 0.82])
            ax.bar(X + 0.00, stats[i, 5, :], color=[0.5, 0.5, 0], width=0.10, label="RDA-OMUNet++")
            ax.bar(X + 0.10, stats[i, 1, :], color='g', width=0.10, label="SGO-OMUNet++")
            ax.bar(X + 0.20, stats[i, 2, :], color=[0, 0.5, 0.5], width=0.10, label="TOA-OMUNet++")
            ax.bar(X + 0.30, stats[i, 3, :], color='m', width=0.10, label="HSO-OMUNet++")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', width=0.10, label="EHSO-OMUNet++")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                       ncol=3, fancybox=True, shadow=True)
            # plt.legend(loc=10)
            # plt.ylim([70, 100])
            path1 = "./Results/Dataset_%s_%s_alg-segmentation.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 4, :], color=[0.5, 0.6, 0], width=0.10, label="UNet")
            ax.bar(X + 0.10, stats[i, 5, :], color='g', width=0.10, label="Res-Unet")
            ax.bar(X + 0.20, stats[i, 6, :], color='m', width=0.10, label="Trans-Unet")
            ax.bar(X + 0.30, stats[i, 7, :], color=[0, 0.6, 0.7], width=0.10, label="MUNet++")
            ax.bar(X + 0.40, stats[i, 8, :], color='k', width=0.10, label="EHSO-OMUNet++")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            # plt.legend(loc=10)
            # plt.ylim([70, 100])
            path1 = "./Results/Dataset_%s_%s_met-segmentation.png" % (str(n + 1), Terms[i - 4])
            plt.savefig(path1)
            plt.show()


def plot_results_optimizer():
    # matplotlib.use('TkAgg')
    eval = np.load('Eval_all.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1-Score', 'MCC']
    Graph_Term = [4, 5, 6, 7, 8]
    Classifier = ['TERMS', 'CNN', 'RAN', 'MANet', 'GRU', 'PROPOSED']
    for i in range(eval.shape[0]):
        value = eval[i, 4, :, 4:]

    for i in range(eval.shape[0]):
        for j in range(len(Graph_Term)):
            Graph = np.zeros((eval.shape[1], eval.shape[2]))
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Term[j] + 4] * 100
            Optimizer = ['Adam', 'SGD', 'RMSProp', 'Adadelta', 'AdaGrad']

            fig = plt.figure()
            # ax = plt.axes(projection="3d")
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 0], color='y', width=0.10, label="CNN")
            ax.bar(X + 0.10, Graph[:, 1], color='g', width=0.10, label="RAN")
            ax.bar(X + 0.20, Graph[:, 2], color='b', width=0.10, label="MANet")
            ax.bar(X + 0.30, Graph[:, 3], color=[0.5, 0.8, 0.8], width=0.10, label="GRU")
            ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="MANet-GRU")
            plt.xticks(X + 0.10,
                       ('Adam', 'SGD', 'RMSProp', 'Ada-delta', 'AdaGrad'))
            plt.xlabel('Optimizer')
            # ax.tick_params(axis='x', labelrotation=45)
            plt.ylabel(Terms[Graph_Term[j]])
            # plt.ylim([60, 100])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/Dataset-%s-%s-bar.png" % (i + 1, Terms[Graph_Term[j]])
            plt.savefig(path1)
            plt.show()


if __name__ == '__main__':
    plot_results_optimizer()
    plot_results()
    plot_Segmentation_results_1()
    plotConvResults()
    Plot_ROC_Curve()
    Image_Results()
    Sample_Images()
