import os
import numpy as np
import warnings
import pandas as pd
from matplotlib import pylab
from sklearn.metrics import roc_curve
from itertools import cycle
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import cv2 as cv
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import pylab

warnings.filterwarnings("ignore")

no_of_dataset = 2


def stats(val):
    v = np.zeros(5)
    v[0] = max(val)
    v[1] = min(val)
    v[2] = np.mean(val)
    v[3] = np.median(val)
    v[4] = np.std(val)
    return v


def plot_conv():
    Dataset = ['Dataset 1', 'Dataset 2', 'Dataset 3']
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['Terms', 'SAA-AMNet-SSDV2', 'ECO-AMNet-SSDV2', 'QSO-AMNet-SSDV2', 'LOA-AMNet-SSDV2',
                 'ILOA-RU-AMNet-SSDV2']
    for i in range(Fitness.shape[0]):
        Terms = ['Worst', 'Best', 'Mean', 'Median', 'Std']
        Conv_Graph = np.zeros((Fitness.shape[1], 5))
        for j in range(len(Algorithm) - 1):
            Conv_Graph[j, :] = stats(Fitness[i, j, :])
        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms)
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
        print('-------------------------------------------------- Statistical Report', str(Dataset[i]),
              ' --------------------------------------------------')

        print(Table)
        length = np.arange(Fitness.shape[-1])
        Conv_Graph = Fitness[i]
        plt.plot(length, Conv_Graph[0, :], color='#e50000', linewidth=3, markersize=12, label=Algorithm[1])
        plt.plot(length, Conv_Graph[1, :], color='#0504aa', linewidth=3, markersize=12, label=Algorithm[2])
        plt.plot(length, Conv_Graph[2, :], color='#76cd26', linewidth=3, markersize=12, label=Algorithm[3])
        plt.plot(length, Conv_Graph[3, :], color='#b0054b', linewidth=3, markersize=12, label=Algorithm[4])
        plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, markersize=12, label=Algorithm[5])
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        plt.savefig("./Results/Convergence_%s.png" % (Dataset[i]))
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('Convergence Curve of ' + str(Dataset[i]))
        plt.show()


def ROC_curve():
    lw = 2
    cls = ['VGG-16', 'Densenet', 'Resnet', 'RAN', 'DRNet-SE']
    for n in range(no_of_dataset):
        Actual = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True).astype('int')
        colors = cycle(["#fe2f4a", "#0165fc", "#00ffff", "lime", "black"])
        for i, color in zip(range(len(cls)), colors):
            Predicted = np.load('Y_Score_' + str(n + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i],
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/ROC_Dataset_%s.png" % (n + 1)
        fig = pylab.gcf()
        fig.canvas.manager.set_window_title('ROC curve')
        plt.savefig(path)
        plt.show()


def Plot_Confusion_():
    for n in range(no_of_dataset):
        Actual = np.load('Actual_' + str(n + 1) + '.npy', allow_pickle=True)
        Predict = np.load('Predict_' + str(n + 1) + '.npy', allow_pickle=True)
        cm = confusion_matrix(np.asarray(Actual).argmax(axis=1), np.asarray(Predict).argmax(axis=1))

        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm)
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_display.plot(ax=ax, cmap='Blues', values_format='d', text_kw={'fontsize': 12})
        ax.set_xlabel('Predicted labels', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual labels', fontsize=12, fontweight='bold')
        ax.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
        if n == 0:
            Classes = [str(i).zfill(3) for i in range(103) if
                       str(i).zfill(3) not in ['031', '060', '061', '064', '076',
                                               '081', ]]  # because these labels are not have annotation
            rot = 45
            fontsizes = 5
            ax.set_xticklabels(Classes, fontsize=fontsizes, rotation=rot)
            ax.set_yticklabels(Classes, fontsize=fontsizes)
        elif n == 1:
            Classes = ['FruitMoth', 'Gall Flies', 'Locust', 'Stem Borer']
            rot = 45
            fontsizes = 10
            ax.set_xticklabels(Classes, fontsize=fontsizes, rotation=rot)
            ax.set_yticklabels(Classes, fontsize=fontsizes)
        else:
            Classes = ['Aphids', 'Armyworm', 'Beetle', 'Bollworm', 'Grasshopper',
                       'Mites', 'Mosquito', 'Sawfly', 'Stem borer']
            rot = 45
            fontsizes = 10
            ax.set_xticklabels(Classes, fontsize=fontsizes, rotation=rot)
            ax.set_yticklabels(Classes, fontsize=fontsizes)
        ax.set_xticklabels(Classes, fontsize=fontsizes, rotation=rot)
        ax.set_yticklabels(Classes, fontsize=fontsizes)
        plt.tight_layout()
        path = "./Results/Confusion_matrix_%s.png" % (n + 1)
        plt.savefig(path)
        plt.show()


def Plot_Confusion():
    Plot_Confusion_()
    results_dir = "./Results"
    os.makedirs(results_dir, exist_ok=True)
    writer = pd.ExcelWriter(os.path.join(results_dir, "Confusion_Matrices.xlsx"), engine='xlsxwriter')
    for n in range(no_of_dataset):
        Actual = np.load(f'Actual_{n + 1}.npy', allow_pickle=True)
        Predict = np.load(f'Predict_{n + 1}.npy', allow_pickle=True)
        actual_labels = np.asarray(Actual).argmax(axis=1)
        predicted_labels = np.asarray(Predict).argmax(axis=1)
        cm = confusion_matrix(actual_labels, predicted_labels)
        if n == 0:
            Classes = [str(i).zfill(3) for i in range(103) if
                       str(i).zfill(3) not in ['031', '060', '061', '064', '076',
                                               '081']]  # because these labels are not have annotation
            rot = 45
            fontsize = 5
        elif n == 1:
            Classes = ['FruitMoth', 'Gall Flies', 'Locust', 'Stem Borer']
            rot = 45
            fontsize = 10
        else:
            Classes = ['Aphids', 'Armyworm', 'Beetle', 'Bollworm', 'Grasshopper',
                       'Mites', 'Mosquito', 'Sawfly', 'Stem borer']
            rot = 45
            fontsize = 10

        df_cm = pd.DataFrame(cm, index=Classes, columns=Classes)
        df_cm.to_excel(writer, sheet_name=f'Dataset_{n + 1}')

        fig, ax = plt.subplots(figsize=(8, 6))
        disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=Classes)
        disp.plot(ax=ax, cmap='Blues', values_format='d', xticks_rotation=rot)
        # ax.set_title(f'Confusion Matrix - Dataset {n + 1}', fontsize=12, fontweight='bold')
        ax.set_title(f'Confusion Matrix', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted Labels', fontsize=12, fontweight='bold')
        ax.set_ylabel('Actual Labels', fontsize=12, fontweight='bold')
        ax.set_xticklabels(Classes, fontsize=fontsize)
        ax.set_yticklabels(Classes, fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f'Confusion_matrix_{n + 1}.png'))
        plt.close()
    writer.close()


def Plot_Hidden_Neurons():
    eval = np.load('Eval_ALL_HN.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'False Positive Rate', 'False Negative Rate',
             'Negative Predictive Value', 'False Discovery Rate', 'F1 score',
             'Matthews Correlation Coefficient', 'False Omission Rate', 'Prevalence Threshold',
             'Critical Success Index', 'Balanced Accuracy', 'Fowlkes–Mallows Index', 'Bookmaker Informedness',
             'Markedness', 'lrplus', 'lrminus', 'Diagnostic Odds Ratio', 'Prevalence']

    Classifiers = ['VGG-16', 'Densenet', 'Resnet', 'RAN', 'DRNet-SE']
    HN = ['20', '40', '60', '80', '100']
    Graph_Terms = [0, 3, 6, 9, 10]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graphs = eval[n, :, :, Graph_Terms[j] + 4]
            Graph = Graphs[4, :]

            methods = np.array([Graph[0], Graph[1], Graph[2], Graph[3], Graph[4]])
            fig, ax = plt.subplots(figsize=(10, 6))
            bar_width = 0.5
            bars = ax.bar(Classifiers, methods, color='lightgreen', edgecolor='black', alpha=0.6, width=bar_width)
            max_height = max(methods)
            padding = max_height * 0.02  # 2% above highest bar
            for bar, count in zip(bars, methods):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,  # Center the text horizontally
                    bar.get_height() + padding,  # Move text 2% above the bar
                    f"{count:.4f}",  # Format text to 4 decimal places
                    ha='center',
                    va='bottom',
                    fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.3'))
            # plt.xlabel('100 Hidden Neuron Count →', fontsize=12, fontweight='bold', color='#35530a')
            plt.ylabel(Terms[Graph_Terms[j]] + ' →', fontsize=12, fontweight='bold', color='#35530a')
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            path = "./Results/HN %s Dataset %s MTD.png" % (Terms[Graph_Terms[j]], n + 1)
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Method Comparison of 100 Hidden Neuron Count vs ' +
                                                Terms[Graph_Terms[j]] + 'dataset' + str(n + 1))
            plt.savefig(path)
            plt.show()


def Plot_EPOCH():
    eval = np.load('Eval_ALL_EP.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Recall', 'Specificity', 'Precision', 'False Positive Rate', 'False Negative Rate',
             'Negative Predictive Value', 'False Discovery Rate', 'F1 score',
             'Matthews Correlation Coefficient', 'False Omission Rate', 'Prevalence Threshold',
             'Critical Success Index', 'Balanced Accuracy', 'Fowlkes–Mallows Index', 'Bookmaker Informedness',
             'Markedness', 'lrplus', 'lrminus', 'Diagnostic Odds Ratio', 'Prevalence']
    Algorithm = np.asarray(['CWO', 'TOT', 'WOA', 'AOA', 'PROPOSED'])
    Classifier = np.asarray(['VGG-16', 'Densenet', 'Resnet', 'RAN', 'DRNet-SE'])
    Epoch = ['20', '40', '60']  # , '80', '100']
    Epoch = np.asarray(Epoch)
    Graph_Terms = [0, 9, 10]
    for n in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = eval[n, :, :, Graph_Terms[j] + 4]

            Mtd_Val = Graph[:3, :]
            bar_width = 0.15  # Width of the bars
            X = np.arange(Mtd_Val.shape[0])  # Positions for bars
            colour = ['#e50000', '#ffff14', '#15b01a', '#0165fc', '#FF6347']
            fig, ax = plt.subplots(figsize=(12, 6))
            bars1 = ax.bar(X + 0.00, Mtd_Val[:, 0], color=colour[0], edgecolor='w', width=0.15, label=Classifier[0])
            for i, bar in enumerate(bars1):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{height:.2f}',
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='square,pad=0.2'))
            bars2 = ax.bar(X + 0.15, Mtd_Val[:, 1], color=colour[1], edgecolor='w', width=0.15, label=Classifier[1])
            for i, bar in enumerate(bars2):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{height:.2f}',
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
            bars3 = ax.bar(X + 0.30, Mtd_Val[:, 2], color=colour[2], edgecolor='w', width=0.15, label=Classifier[2])
            for i, bar in enumerate(bars3):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{height:.2f}',
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
            bars4 = ax.bar(X + 0.45, Mtd_Val[:, 3], color=colour[3], edgecolor='w', width=0.15, label=Classifier[3])
            for i, bar in enumerate(bars4):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{height:.2f}',
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
            bars5 = ax.bar(X + 0.60, Mtd_Val[:, 4], color=colour[4], edgecolor='w', width=0.15, label=Classifier[4])
            for i, bar in enumerate(bars5):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2, height / 2, f'{height:.2f}',
                        ha='center', va='center', fontsize=10, fontweight='bold',
                        bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))

            ax.set_xticks(X + (len(Classifier) - 1) * bar_width / 2)
            ax.set_xticklabels(Epoch, rotation=0, fontsize=9, fontname="Arial", fontweight='bold')
            ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
            ax.set_ylabel(Terms[Graph_Terms[j]], fontsize=12, fontweight='bold')
            plt.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
            ymax = np.max(Mtd_Val)
            step = 0.05 if ymax <= 1 else (ymax + 2) / 10
            plt.yticks(np.arange(0, ymax + step, step))
            plt.legend(loc='center left', bbox_to_anchor=(0.95, 0.9), fontsize=10, labelspacing=1, handlelength=1)
            ax.spines['top'].set_color('lightgray')
            ax.spines['top'].set_linewidth(0.0)
            ax.spines['right'].set_color('lightgray')
            ax.spines['right'].set_linewidth(0.0)
            path = "./Results/Dataset_%s_Epoch_%s_Mtd.png" % (n + 1, Terms[Graph_Terms[j]])
            plt.tight_layout()
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title(
                'Epoch vs ' + Terms[Graph_Terms[j]] + ' Method Comparision of Dataset' + str(n + 1))
            plt.savefig(path)
            plt.show()


def Plot_Batch_size():
    eval = np.load('Eval_ALL_BS.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'False Positive Rate', 'False Negative Rate',
             'Negative Predictive Value', 'False Discovery Rate', 'F1 score',
             'Matthews Correlation Coefficient', 'False Omission Rate', 'Prevalence Threshold',
             'Critical Success Index', 'Balanced Accuracy', 'Fowlkes–Mallows Index', 'Bookmaker Informedness',
             'Markedness', 'lrplus', 'lrminus', 'Diagnostic Odds Ratio', 'Prevalence']
    # Table_Term = [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 13]
    Table_Term = [0, 8, 13]
    Batch_Size = ['Batch Size 4', 'Batch Size 8', 'Batch Size 16', 'Batch Size 32', 'Batch Size 64']
    Classifier = ['Batch Size', 'VGG-16', 'Densenet', 'Resnet', 'RAN', 'DRNet-SE']
    for n in range(eval.shape[0]):
        for k in range(len(Table_Term)):
            value = eval[n, :5, :, Table_Term[k] + 4]
            Table = PrettyTable()
            Table.add_column(Classifier[0], Batch_Size)
            for j in range(len(Classifier) - 1):
                Table.add_column(Classifier[j + 1], value[:, j])
            print('--------------------------------------------------', str(Terms[Table_Term[k]]),
                  ' Batch Size vs Classifier Comparison Dataset ' + str(
                      n + 1) + ' --------------------------------------------------')
            print(Table)


def Plot_Seg_Results():
    Eval = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC']
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN']  # , 'STD']
    Full = ['TERMS', 'SAA-AMNet-SSDV2', 'ECO-AMNet-SSDV2', 'QSO-AMNet-SSDV2', 'LOA-AMNet-SSDV2', 'ILOA-RU-AMNet-SSDV2',
            'CNN', 'Faster-RCNN', 'YoloV3', 'MNet-SSDV2', 'ILOA-RU-AMNet-SSDV2']
    Graph_terms = [0, 1, 2, 3, 7, 12]
    for n in range(Eval.shape[0]):
        stats = np.zeros((len(Graph_terms), Eval.shape[-3] + 1, 5))  # (METRICS, ALGORITHM, STATS)
        Eval_all = Eval[n]
        for k in range(len(Graph_terms)):
            for j in range(Eval_all.shape[-3] + 1):
                if j < Eval_all.shape[-3]:
                    stats[k, j, 0] = np.max(Eval_all[j][:, Graph_terms[k] + 4])
                    stats[k, j, 1] = np.min(Eval_all[j][:, Graph_terms[k] + 4])
                    stats[k, j, 2] = np.mean(Eval_all[j][:, Graph_terms[k] + 4])
                    stats[k, j, 3] = np.median(Eval_all[j][:, Graph_terms[k] + 4])
                    stats[k, j, 4] = np.std(Eval_all[j][:, Graph_terms[k] + 4])

            alg_prop = stats[k, 4, :]
            stats[k, 9, :] = alg_prop

            Table = PrettyTable()
            Table.add_column(Full[0], ['STD'])
            for l in range(len(Full) - 6):
                Table.add_column(Full[l + 1], stats[k, l, -1:])
            print('-------------------------------------------------- Standard deviation of ' + Terms[
                Graph_terms[k]] + ' Algorithm Comparison of Dataset',
                  str(n + 1),
                  '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Full[0], ['STD'])
            for l in range(5, 10):
                Table.add_column(Full[l + 1], stats[k, l, -1:])
            print('-------------------------------------------------- Standard deviation of ' + Terms[
                Graph_terms[k]] + ' Classifier Comparison of Dataset',
                  str(n + 1),
                  '--------------------------------------------------')
            print(Table)

            # Algorithm comparision
            Alg_Val = stats[k, :5, :-1]
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(Alg_Val.shape[-1])
            X = x
            colour = ['#00BFC4', '#FF9224', '#B5D334', '#9370DB', '#FF6347']
            bar_width = 0.15
            offsets = [0.00, 0.15, 0.30, 0.45, 0.60]
            bars = []
            for i in range(5):
                bar_group = ax.bar(X + offsets[i], Alg_Val[i, :], color=colour[i], edgecolor='w', width=bar_width)
                bars.append(bar_group)
                highlight_map = {
                    0: [0, 4],  # BEST → 1st and 5th
                    1: [1],  # WORST → 2nd
                    2: [2],  # MEAN → 3rd
                    3: [3],  # MEDIAN → 4th
                }
                for j in range(len(X)):  # Loop over statistics (BEST, WORST, etc.)
                    bar_height = Alg_Val[i, j]
                    x_center = X[j] + offsets[i]
                    # y_top = bar_height + 10  # 1.5
                    y_top = bar_height * 1.20
                    label_text = Full[i + 1]
                    if i in highlight_map.get(j, []):  # Check if this algorithm should be labeled in this stat

                        ax.text(x_center + 0.125, y_top + 1.25, label_text, ha='center', va='bottom',  # 1.5
                                fontsize=10, fontweight='bold')
                        ax.hlines(y=y_top, xmin=x_center, xmax=x_center + 0.21,
                                  color=colour[i], linewidth=2.5)
                        ax.vlines(x=x_center, ymin=bar_height, ymax=y_top,
                                  color='k', linestyle='dotted', linewidth=1.5)  # color=colour[i]

            ax.set_xticks(X + (bar_width * 2))
            ax.set_xticklabels(Statistics, fontsize=10, fontweight='bold')
            ax.set_xlabel('Statistical Analysis', fontsize=12, fontweight='bold')
            ax.set_ylabel(Terms[Graph_terms[k]], fontsize=12, fontweight='bold')
            plt.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
            plt.yticks(np.arange(0, np.max(Alg_Val) + 7, (np.max(Alg_Val) + 2) / 10))
            ax.spines['top'].set_color('lightgray')
            ax.spines['top'].set_linewidth(0.0)
            ax.spines['right'].set_color('lightgray')
            ax.spines['right'].set_linewidth(0.0)
            plt.tight_layout()
            plt.savefig(f"./Results/Seg_{Terms[Graph_terms[k]]}_Dataset_{n + 1}_Alg.png")
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title(
                'Algorithm comparision of Statistical Analysis vs ' + Terms[
                    Graph_terms[k]] + ' Algorithm Comparision of Dataset' + str(n + 1))
            plt.show()

            # Method comparision
            Mtd_Val = stats[k, 5:, :-1]
            fig, ax = plt.subplots(figsize=(12, 6))
            x = np.arange(Mtd_Val.shape[-1])
            X = x
            colour = ['#00BFC4', '#FF9224', '#B5D334', '#9370DB', '#FF6347']
            bar_width = 0.15
            offsets = [0.00, 0.15, 0.30, 0.45, 0.60]
            bars = []
            for i in range(5):
                bar_group = ax.bar(X + offsets[i], Mtd_Val[i, :], color=colour[i], edgecolor='w', width=bar_width)
                bars.append(bar_group)

                highlight_map = {
                    0: [0, 4],  # BEST → 1st and 5th
                    1: [1],  # WORST → 2nd
                    2: [2],  # MEAN → 3rd
                    3: [3],  # MEDIAN → 4th
                }
                for j in range(len(X)):  # Loop over statistics (BEST, WORST, etc.)
                    bar_height = Mtd_Val[i, j]
                    x_center = X[j] + offsets[i]
                    y_top = bar_height * 1.20
                    # y_top = bar_height + 10  # 1.5
                    label_text = Full[i + 6]
                    if i in highlight_map.get(j, []):
                        ax.text(x_center + 0.125, y_top + 1.25, label_text, ha='center', va='bottom',  # 1.5
                                fontsize=10, fontweight='bold')
                        ax.hlines(y=y_top, xmin=x_center, xmax=x_center + 0.21,
                                  color=colour[i], linewidth=2.5)
                        ax.vlines(x=x_center, ymin=bar_height, ymax=y_top,
                                  color='k', linestyle='dotted', linewidth=1.5)  # color=colour[i]
            ax.set_xticks(X + (bar_width * 2))
            ax.set_xticklabels(Statistics, fontsize=10, fontweight='bold')
            ax.set_xlabel('Statistical Analysis', fontsize=12, fontweight='bold')
            ax.set_ylabel(Terms[Graph_terms[k]], fontsize=12, fontweight='bold')
            plt.grid(axis='y', which='major', linestyle='--', linewidth=0.7, alpha=0.7)
            plt.yticks(np.arange(0, np.max(Alg_Val) + 7, (np.max(Alg_Val) + 2) / 10))
            ax.spines['top'].set_color('lightgray')
            ax.spines['top'].set_linewidth(0.0)
            ax.spines['right'].set_color('lightgray')
            ax.spines['right'].set_linewidth(0.0)
            plt.tight_layout()
            plt.savefig(f"./Results/Seg_{Terms[Graph_terms[k]]}_Dataset_{n + 1}_Mtd.png")
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title(
                'Statistical Analysis vs ' + Terms[Graph_terms[k]] + ' Method Comparision of Dataset' + str(n + 1))
            plt.show()


def Plot_Epoch_Seg_Results():
    Eval = np.load('Eval_all_SPE.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'PSNR', 'MSE', 'Recall', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV', 'FDR', 'F1 Score', 'MCC']
    Graph_terms = [0, 1, 2, 3, 7, 12]
    Classifier = ['MNet-SSDV2', 'ILOA-RU-AMNet-SSDV2']
    SPE = ['20', '40', '60', '80', '100']
    for n in range(Eval.shape[0]):
        stats = np.zeros((Eval.shape[-4], len(Graph_terms), Eval.shape[-3], 5))
        for p in range(Eval.shape[-4]):
            Eval_all = Eval[n, p]
            for i in range(len(Graph_terms)):
                for j in range(Eval_all.shape[-3]):
                    values = Eval_all[j][:, Graph_terms[i] + 4]
                    stats[p, i, j, 0] = np.max(values)
                    stats[p, i, j, 1] = np.min(values)
                    stats[p, i, j, 2] = np.mean(values)
                    stats[p, i, j, 3] = np.median(values)
                    stats[p, i, j, 4] = np.std(values)

        for k in range(len(Graph_terms)):
            alg_prop = stats[:, :, 4, :]
            stats[:, :, 9, :] = alg_prop  # duplicate entry for index matching
            Graph = stats[:, k, :, 2]  # MEAN values for current metric

            values = np.asarray([Graph[:, 8], Graph[:, 4]])  # RViT_ARDDNet, PROPOSED
            fig, ax = plt.subplots(figsize=(8, 6))

            bar_width = 0.25  # narrower bars
            group_spacing = 0.2  # space between groups
            bar_spacing = 0.05  # space between bars within a group
            x = np.arange(len(SPE)) * (2 * bar_width + bar_spacing + group_spacing)

            colors = ['#F8766D', '#00BFC4']
            for i in range(len(values)):
                # ax.bar(x + k * bar_width, values[k], width=bar_width, label=Classifier[k], color=colors[k])
                bar_x = x + i * (bar_width + bar_spacing)
                ax.bar(bar_x, values[i], width=bar_width, label=Classifier[i], color=colors[i])

            ax.grid(True, which='major', axis='y', color='k', linestyle='-', linewidth=0.5)
            ax.set_axisbelow(True)
            ax.set_xlabel('Epoch', fontsize=12)
            ax.set_ylabel(Terms[Graph_terms[k]], fontsize=12)
            ax.set_xticks(x + bar_width / 2)
            ax.set_xticklabels(SPE)
            from matplotlib.lines import Line2D
            custom_legend = [Line2D([0], [0], marker='s', color='w', label=Classifier[i],
                                    markerfacecolor=colors[i], markersize=10) for i in range(len(Classifier))]
            plt.legend(handles=custom_legend, fontsize=12, loc='upper center', bbox_to_anchor=(0.5, -0.1),
                       frameon=False, ncol=5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            plt.tight_layout()
            fig = pylab.gcf()
            fig.canvas.manager.set_window_title('Detection (Mean)  vs ' + Terms[Graph_terms[k]] +
                                                ' Method Comparision of Dataset' + str(n + 1))
            path = "./Results/Epoch_Seg_%s_Dataset_%s_Mean_Mtd.png" % (Terms[Graph_terms[k]], str(n + 1))
            plt.savefig(path)
            plt.show()


def Image_Results():
    for n in range(no_of_dataset):
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        Segmented = np.load('Detected_' + str(n + 1) + '.npy', allow_pickle=True)
        # index = [450, 500, 2000, 2500, 3000]
        index = [150, 600, 700, 800, 1000]
        for i in range(len(index)):
            print(n, no_of_dataset, index[i], 5)
            image = cv.resize(Images[index[i]], (512, 512))
            Seg = cv.resize(Segmented[index[i]], (512, 512))
            plt.suptitle("Image %d from dataset %d" % (i + 1, n + 1), fontsize=20)
            plt.subplot(1, 2, 1)
            plt.title('Original Image')
            plt.imshow(image)
            plt.subplot(1, 2, 2)
            plt.title('Detected Image')
            plt.imshow(Seg)
            path1 = "./Results/Image_Results/Dataset_%s_image_%s.png" % (n + 1, i + 1)
            plt.savefig(path1)
            plt.show()
            cv.imwrite('./Results/Image_Results/Orig_' + str(i + 1) + '_Dataset_-' + str(n + 1) + '.png', image)
            cv.imwrite('./Results/Image_Results/Pest_' + str(i + 1) + '_Dataset_-' + str(n + 1) + '.png', Seg)


def Sample_images():
    for n in range(no_of_dataset):
        Images = np.load('Images_' + str(n + 1) + '.npy', allow_pickle=True)
        Target = np.load('Target_' + str(n + 1) + '.npy', allow_pickle=True)
        if Target.shape[-1] >= 2:
            targ = np.argmax(Target, axis=1).reshape(-1, 1)
        else:
            targ = Target
        class_indices = {}
        for class_label in np.unique(targ):
            indices = np.where(targ == class_label)[0]
            class_indices[class_label] = indices
        for class_label, indices in class_indices.items():
            if n == 0:
                labels = [str(i).zfill(3) for i in range(103) if
                          str(i).zfill(3) not in ['031', '060', '061', '064', '076',
                                                  '081']]  # because these labels are not have annotation
            elif n == 1:
                labels = ['FruitMoth', 'Gall Flies', 'Locust', 'Stem Borer']
            else:
                labels = ['Aphids', 'Armyworm', 'Beetle', 'Bollworm', 'Grasshopper',
                          'Mites', 'Mosquito', 'Sawfly', 'Stem borer']
            if len(indices) >= 5:
                no_samples = 5
            else:
                no_samples = len(indices)
            for i in range(no_samples):
                print(n, no_of_dataset, labels[class_label], i)
                Image = cv.resize(Images[indices[i]], (512, 512))
                cv.imshow('Image', Image)
                cv.waitKey(750)
                cv.imwrite('./Results/Sample_Images/Dataset_' + str(n + 1) + '_' + str(
                    labels[class_label]) + '_image_' + str(i + 1) + '.png', Image)


if __name__ == '__main__':
    plot_conv()
    ROC_curve()
    Plot_Confusion()
    Plot_Hidden_Neurons()
    Plot_EPOCH()
    Plot_Batch_size()
    Plot_Seg_Results()
    Plot_Epoch_Seg_Results()
    Image_Results()
    Sample_images()
