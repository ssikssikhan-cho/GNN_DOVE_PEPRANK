# Copyright 2024 Joon-Sang Park. All Rights Reserved.

import sys
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

evalset = ['1J2X_A_18_B', '1GUX_B_9_E', '1LB6_A_9_B', '1OAI_A_9_B', '1SQK_A_25_B', '1WKW_A_20_B', '2G30_A_16_P', '2HPL_A_5_B', '2NM1_A_17_B', '2OKR_A_24_C', '2PEH_A_10_C', '2PUY_B_10_E', '2PV1_A_7_B', '2QN6_A_18_C', '2QOS_C_11_A', '3D9U_A_6_B', '3H8A_A_28_E', '3HBV_P_7_Z', '3LU9_B_25_C', '3MHP_A_26_C', '3O37_A_10_E', '3PLV_A_21_C', '4K0U_A_15_B', '4M5S_A_10_B', '4Q5U_A_24_C', '4QJA_A_10_P', '5CRW_A_11_B', '5EPP_A_15_B', '5FZT_A_23_B']

#3RYB_A_9_B, 1UHB_A_9_P zero crrect decoy

# function find out if there is a file corresponding to given pattern in the directory and return true if there is
def find_file(directory, pattern):
    for file in os.listdir(directory):
        if pattern in file:
            return True
    return False

def prepinferece(basepath):
    curdir = os.getcwd()
    for index, elem in enumerate(evalset):
        infpath = os.path.join(basepath, elem)
        os.chdir(infpath)
        os.system(f'echo {infpath}; ls fixed-reduce | wc -l')
        if find_file(infpath, 'correctlist') == False:
            os.system('awk -F\',\' \'{if ($2 >= 0.8) print $1}\' dockQ_results_sorted.csv > correctlist')
            #os.system('ls correctlist; cat correctlist')
        if find_file(os.path.join(infpath, 'fixed-reduce'), 'crt') == True:
            print(f'crt found in {infpath}'+"/fixed-reduce")
            continue
        defiles=[x for x in os.listdir(infpath) if ".pdb" in x and not "-if" in x]
        for index, file in enumerate(defiles):
            if os.system(f'grep {file} correctlist') == 0:
                os.system(f'mv {file} {file[:-3]}crt.pdb')
        os.system(f'mkdir fixed-reduce')
        os.system('ls --hide=\'*-if.pdb\' --hide=\'*csv\' --hide=\'*list\' | xargs -I{} sh -c \'reduce {} > fixed-reduce/{}\'')
        os.system('ls *crt.pdb | xargs -I{} sh -c \'reduce {} > fixed-reduce/{}\'')


def genpredictedscores(basepath, parampath):
    curdir = os.getcwd()
    for index, elem in enumerate(evalset):
        infpath = os.path.join(basepath, elem)
        os.system(f'python inference.py -F {infpath}/fixed-reduce/ --parampath {parampath} --gpu \'2\'')
        os.system(f'mv inf_results/fixed-reduce ' + os.path.join('inf_results', elem))


if __name__ == '__main__':
    #baseinfapath = '/rv2/biodata/pep_dataset/'
    paramfilepath = '/home2/escho/GNN_DOVE_PEPRANK/model/2024-11-19T12:54.pth.tar'
    if (len(sys.argv) > 2):
        baseinfpath = sys.argv[1]
    if (len(sys.argv) > 3):    
        paramfilepath = sys.argv[2]

#    genpredictedscores(baseinfapath, paramfilepath)
#    exit()
    graph_xmax = 1000

    #curdir = os.getcwd()
    #infpath = os.path.join(curdir, 'inf_results/middle/')
    infpath="/home2/escho/GNN_DOVE_PEPRANK/inf_results/"
    results=[x for x in os.listdir(infpath) if os.path.isdir(os.path.join(infpath, x))]
    n_datasets = len(results)
    print(f"Number of datasets: {n_datasets}")

    # the board
    board = [[0] * graph_xmax for _ in range(n_datasets)]
    board2 = [[0] * graph_xmax for _ in range(n_datasets)]
    
    """
    for i in range(n_datasets):
        sor = os.path.join(infpath, results[i], 'predictions_sorted.txt')
        print(f"Processing file: {sor}")

        try:
            with open(sor, 'r') as file:
                lines = file.read().splitlines()
        except FileNotFoundError:
            print(f"File not found: {sor}")
            continue

        lines = [line for line in lines if not line.startswith('Input') and line.strip()]

        samesame = 0
        previous = None

        for k in range(1, graph_xmax):
            if k >= len(lines):
                print(f"Skipping index {k}, out of range.")
                continue

            parts = lines[k].split('\t')
            if len(parts) > 1:
                score = float(parts[1])
                if previous is None or score != previous:
                    samesame += 1
                previous = score
                board2[i][k] = samesame
            else:
                print(f"Skipping invalid line: {lines[k]}")

        for k in range(graph_xmax):
            if k >= len(lines):
                continue

            parts = lines[k].split('\t')
            if len(parts) > 1:
                score = float(parts[1])
                name = parts[0]

                if 'crt' in name:
                    print(f"Found 'crt' in name: {name}, score: {score}, rank: {k}")

                    for j in range(graph_xmax):
                        if board2[i][k] == board2[i][j]:
                            ind = board2[i][j]
                            print(f"Match at index {j} with rank {ind}")
                            break

                    for c in range(ind, graph_xmax):
                        board[i][c] = 1

                    break

    print(f"Board2[0]: {board2[0]}")
    print(f"Board[0]: {board[0]}")

    b = np.array(board)
    c = np.sum(b, axis=0) / n_datasets

    x = np.arange(0, graph_xmax, 5)

    plt.figure(linewidth=5)
    plt.xlim(-0.5, graph_xmax - 1)
    plt.xticks(x)
    plt.ylim(0, 1.05)
    plt.plot(c, color='r', label='GNN_EA w/ 71K dataset')

    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.xlabel("Top Rank Considered")
    plt.ylabel("Hit Rate")
    plt.title("Train Code validation TT")
    plt.legend()

    area = np.trapz(c, dx=1)
    print(f"Area under the curve: {area}")
    plt.text(0.5, 0.8, f'Area: {area:.2f}', fontsize=12, color='black', ha='center')
    plt.savefig('/home2/escho/GNN_DOVE_PEPRANK/inf_results/middle/hitrate_ea.png')

    """
    # # for every txt files
    for i in range(n_datasets):
        
        sor = os.path.join(infpath, results[i], 'predictions_sorted.txt')
        print("sor: ",sor)

        with open(sor,'r') as file:
            lines = file.read().split('\n')

        lines = [line for line in lines if not line.startswith('Input') and line.strip()]
        
        samesame = 0
        previous = float(lines[0].split('\t')[1])
        ind = 0
        
        for k in range(1, graph_xmax):
            if len(lines[k].split('\t')) > 1:
                score = float(lines[k].split('\t')[1])
                #print('previous: ', previous)
                #print('score: ', score)
                if score != previous:
                    samesame += 1
                previous = score
                board2[i][k] = samesame
        #print(f'board2[{i}]: ', board[i])
            else:
                print(f"Skipping invalid line (not enough columns): {lines[k]}")
                continue  # 두 번째 항목이 없는 경우 건너뛰기
        for k in range(graph_xmax):
            if len(lines[k].split('\t')) > 1:
                score = float(lines[k].split('\t')[1])
                name = lines[k].split('\t')[0]
                    
                # if there is one hit in group, after that rank is always hit.
                if 'crt' in name:
                    print(name, score, k)
                    # find the first same rank
                    for j in range(1,graph_xmax):
                        if board2[i][k] == board2[i][j]:
                            # print(board2[i][k] , board2[i][j])
                            ind = board2[i][j]
                            print('ind: ', ind)
                            break
                    # print("the index: ", ind)
                    # first rank to ~
                    for c in range(ind, graph_xmax):
                        board[i][c] = 1
                    
                    break
            else:
                print(f"Skipping invalid line (not enough columns): {lines[k]}")
                continue  # 두 번째 항목이 없는 경우 건너뛰기
        #print(f'board[{i}]: ', board[i])

    print(board2[0])
    print(board[0])
    b = np.array(board)
    c = np.sum(b, axis=0)
    print('c: ', c)
    c = c/n_datasets
    x = np.arange(0,graph_xmax,5)
    print(c)
    plt.figure(linewidth = 5)
    plt.xlim(-0.5, graph_xmax-1)
    plt.xticks(x)
    plt.ylim(0,1.05)
    plt.plot(c, color = 'r', label = 'GNN_EA w/ 71K dataset')
    #plt.plot(e, color = 'b', label = 'atom')
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10))

    plt.xlabel("Top Rank Considered")
    plt.ylabel("Hit Rate")
    plt.title("Train Code validation TT")
    plt.legend()

    # 그래프의 선 아래 면적 계산
    area = np.trapz(c, dx=1)
    print(f"Area under the curve: {area}")
    plt.text(0.5, 0.8, f'Area: {area:.2f}', fontsize=12, color='black', ha='center')
    plt.savefig('/home2/escho/GNN_DOVE_PEPRANK/inf_results/middle/hitrate_ea.png')
