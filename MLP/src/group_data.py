import numpy as np
import csv
import pandas as pd
##false
def false():

    with open("../dataSet/train_data/false_all.csv", 'w',newline='') as f:
        writer = csv.writer(f)
        with open("../dataSet/train_data/false/false0_label_line.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/false/false1_label_line.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/false/false2_label_line.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/false/false3_label_line.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/false/false4_label_line.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/false/false5_label_line.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/false/false6_label_line.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/false/false7_label_line.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/false/false8_label_line.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/false/false9_label_line.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/false/false10_label_line.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)


    # df = pd.read_csv("../dataSet/facade0_all.csv", header=None)
    #
    # info = df.values
    # X = info[:,:-1]
    # Y = info[:,-1:]
    # print(X)
# false()


##True
def true():
    with open("../dataSet/train_data/true_all.csv", 'w',newline='') as f:
        writer = csv.writer(f)
        with open("../dataSet/train_data/true/sshh_true_0.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/true/sshh_true_1.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/true/sshh_true_2.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/true/sshh_true_3.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/true/sshh_true_4.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/true/sshh_true_5.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/true/sshh_true_6.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/true/sshh_true_7.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/true/sshh_true_8.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/true/sshh_true_9.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

        with open("../dataSet/train_data/true/sshh_true_10.txt", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split('\t')
                line_info = line_info[1:6] + line_info[7:12] + line_info[13:18] + line_info[19:-1]
                writer.writerow(line_info)

# true()


def all():
    with open("../dataSet/train_data/trainData.csv", 'w',newline='') as af:
        writer = csv.writer(af)
        with open("../dataSet/train_data/false_all.csv", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split(',')
                writer.writerow(line_info)

        with open("../dataSet/train_data/true_all.csv", 'r') as rf:
            for line in rf:
                line_info = line.strip('\n').split(',')
                writer.writerow(line_info)

# all()