import pandas as pd
import numpy as np
from argparse import ArgumentParser
import h5py


# 序列长度
L = 5000

# N, A, C, G, T
BASE = np.array([
    [0, 0, 0, 0],
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
], dtype=np.int8)


BASE_INDEX = {
    'N': '0',
    'A': '1',
    'C': '2',
    'G': '3',
    'T': '4'}


# Neither, Acceptor, Donor, Padding
SPLICE = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [0, 0, 0]
], dtype=np.int8)


SPLICE_INDEX = {
    'NEITHER': 0,
    'ACCEPTOR': 1,
    'DONOR': 2,
    'PADDING': -1}


def seq_to_numeric(seq):
    seq = seq.upper().replace('A', BASE_INDEX['A']).replace('C', BASE_INDEX['C']).replace('G', BASE_INDEX['G']).replace('T', BASE_INDEX['T']).replace('N', BASE_INDEX['N'])
    return seq


def dna_reverse_complement(seq):
    seq = seq.upper().translate(seq.maketrans('ATCGN', 'TAGCN'))[::-1]
    return seq


def run(args):
    h5f = h5py.File(args.out, 'w')
    x_group = h5f.create_group('X')
    y_group = h5f.create_group('Y')
    # 读取数据集
    df = pd.read_csv(args.dataset, sep="\t", header=None, names=['gene', 'paralog', 'chrom', 'strand', 'tx_start', 'tx_end', 'jn_start', 'jn_end', 'seq'])
    chroms = ['chr2', 'chr4', 'chr6', 'chr8', 'chr10', 'chr11', 'chr12', 'chr13', 'chr14', 'chr15', 'chr16', 'chr17', 'chr18', 'chr19', 'chr20', 'chr21', 'chr22', 'chrX', 'chrY'] if args.mode == "train" else ['chr1', 'chr3', 'chr5', 'chr7', 'chr9']
    for row in df.itertuples():
        if row.chrom not in chroms:
            continue
        # 基因组边缘的跳过
        if row.tx_end - row.tx_start + 10001 != len(row.seq):
            print(f'{row.gene} seq length error')
            continue
        # slop 5k提取基因序列，基因左右5k使用N填充，使得基因的序列达到15k，不足的后面再右侧继续填充
        seq = 'N' * L + row.seq[L: -L] + 'N' * L
        # strand为-的基因序列进行反向互补
        if row.strand == '-':
            seq = dna_reverse_complement(seq)
        # 转成数字方便后续转成多维数组
        seq = seq_to_numeric(seq)
        seq = np.array(list(seq), dtype=np.int8)
        # 序列对应的剪接位点转成数字, 注意jn_start和jn_end是exon维度的边界，相邻外显子的坐标，donor和acceptor为一组
        splice = np.zeros(row.tx_end - row.tx_start + 1, dtype=np.int8)
        jn_start = [int(i) for i in row.jn_start.split(',')[:-1]]
        jn_end = [int(i) for i in row.jn_end.split(',')[:-1]]
        if row.strand == '+':
            for js in jn_start:
                if row.tx_start <= js <= row.tx_end:
                    splice[js - row.tx_start] = SPLICE_INDEX['DONOR']
            for je in jn_end:
                if row.tx_start <= je <= row.tx_end:
                    splice[je - row.tx_start] = SPLICE_INDEX['ACCEPTOR']
        elif row.strand == '-':
            for js in jn_start:
                if row.tx_start <= js <= row.tx_end:
                    splice[row.tx_end - js] = SPLICE_INDEX['ACCEPTOR']
            for je in jn_end:
                if row.tx_start <= je <= row.tx_end:
                    splice[row.tx_end - je] = SPLICE_INDEX['DONOR']
        else:
            raise Exception("strand error")
        # 序列和splice信息右侧填充5k，确保每个基因长度都够15k，后面再以15k长度，5k窗口滑动创建单元数据集
        seq = np.pad(seq, (0, L), 'constant', constant_values=int(BASE_INDEX['N']))
        splice = np.pad(splice, (0, L), 'constant', constant_values=SPLICE_INDEX['PADDING'])
        # 一个基因能构建的单元数据集数量
        nums = np.ceil((row.tx_end - row.tx_start + 1) / L)
        X_genes = []
        Y_genes = []
        # 滑动创建单元数据集
        for i in range(int(nums)):
            x = seq[i * L: i * L + 3 * L]
            y = splice[i * L: i * L + L]
            X = BASE[x]
            Y = SPLICE[y]
            X_genes.append(X)
            Y_genes.append(Y)
        x_group.create_dataset(f'{row.gene}', data=np.array(X_genes))
        y_group.create_dataset(f'{row.gene}', data=np.array(Y_genes))
    h5f.close()


def main():
    parser = ArgumentParser()
    parser.add_argument("-dataset", help="input dataset file")
    parser.add_argument("-mode", help="train or test")
    parser.add_argument("-out", help="output h5 file")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
