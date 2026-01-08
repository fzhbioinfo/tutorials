from pkg_resources import resource_filename
import pandas as pd
import numpy as np
from pyfaidx import Fasta
import logging
from typing import Tuple
from tensorflow.keras.models import load_model


LOGGER = logging.getLogger()
FORMATTER = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
LOGGER.setLevel(logging.INFO)
HANDlER = logging.StreamHandler()
HANDlER.setFormatter(FORMATTER)
LOGGER.addHandler(HANDlER)


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
    'T': '4',
}

class Annotator:
    def __init__(self, fasta_file: str, annotations: str):
        # 基因组序列
        self.fasta = Fasta(fasta_file)
        # 染色体用于判断名称是否有chr前缀
        self.chrom = list(self.fasta.keys())[0]
        # 基因组注释版本
        self.annotations = 'grch38' if '38' in annotations else 'grch37'
        # 导入模型
        self.models = [
            load_model(resource_filename(__name__, f'models/spliceai{i}.h5'))
            for i in range(1, 6)
        ]
        # 读取基因注释文件
        df = pd.read_csv(resource_filename(__name__, f'annotations/{self.annotations}.txt'), sep='\t', dtype={'CHROM': str})
        df['EXON_START'] = df['EXON_START'].str.replace(',$', '', regex=True).str.split(',')
        df['EXON_END'] = df['EXON_END'].str.replace(',$', '', regex=True).str.split(',')
        # 获取基因详细信息转成数组，将0 base转成1 base
        self.names = df['#NAME'].to_numpy()
        self.chroms = df['CHROM'].to_numpy()
        self.strands = df['STRAND'].to_numpy()
        self.tx_starts = df['TX_START'].to_numpy() + 1
        self.tx_ends = df['TX_END'].to_numpy()
        self.exon_starts = [np.array(i, dtype=int) + 1 for i in df['EXON_START'].to_numpy()]
        self.exon_ends = [np.array(i, dtype=int) for i in df['EXON_END'].to_numpy()]

    def get_gene_and_strand(self, chrom: str, pos: int) -> Tuple[np.array, np.array, np.array]:
        # 获取变异位置所在注释文件中的位置索引、基因名称、方向
        chrom = normalise_chrom(chrom, self.chroms[0])
        idxs = np.nonzero(np.logical_and(self.chroms == chrom, np.logical_and(self.tx_starts <= pos, self.tx_ends >= pos)))[0]
        genes = self.names[idxs]
        strands = self.strands[idxs]
        return idxs, genes, strands
    
    def get_boundary_distance(self, idx: int, pos: int) -> Tuple[int, int, np.array]:
        # 获取变异位置的基因/外显子边界距离
        dis_tx_start = pos - self.tx_starts[idx]
        dis_tx_end = self.tx_ends[idx] - pos
        dis_exon = np.hstack([self.exon_starts[idx] - pos, self.exon_ends[idx] - pos])
        return dis_tx_start, dis_tx_end, dis_exon


def normalise_chrom(source: str, target: str) -> str:
    # 染色体名称格式统一
    if target.startswith('chr'):
        if not source.startswith('chr'):
            source = f'chr{source}'
    else:
        if source.startswith('chr'):
            source = source.replace('chr', '')
    return source


def one_hot_encode(seq: str) -> np.array:
    # 序列one-hot编码
    seq = seq.upper().replace('N', BASE_INDEX['N']).replace('A', BASE_INDEX['A']).replace('C', BASE_INDEX['C']).replace('G', BASE_INDEX['G']).replace('T', BASE_INDEX['T'])
    seq = np.array(list(seq), dtype=np.int8)
    seq = BASE[seq]
    return seq

def get_delta_scores(record, ann: Annotator, distance: int, is_mask: bool) -> list:
    # 变异位置以及左右distance长度的序列进行预测
    cov = 2 * distance + 1
    # 神经网络输入序列长度
    seq_len = 10000 + cov
    seq_len_half = seq_len // 2
    # 预测结果默认值
    delta_scores = []
    # 判断变异是否在基因上
    chrom = normalise_chrom(record.chrom, ann.chrom)
    idxs, genes, strands = ann.get_gene_and_strand(chrom, record.pos)
    if idxs.size == 0:
        return delta_scores
    # 目标位置上下游序列
    seq = str(ann.fasta.get_seq(chrom, record.pos - seq_len_half, record.pos + seq_len_half))
    # 跳过序列错误的变异，防止参考基因组与vcf版本不一致
    len_ref = len(record.ref)
    if seq[seq_len_half: seq_len_half + len_ref].upper() != record.ref.upper():
        LOGGER.warning(f'Skipping ref issue: {record.chrom}-{record.pos}-{record.ref}-{record.alts}')
        return delta_scores
    # 构建神经网络输入序列
    for i, idx in enumerate(idxs):
        # 变异距基因和最近的外显子边界距离
        dis_tx_start, dis_tx_end, dis_exon = ann.get_boundary_distance(idx, record.pos)
        # 观测参考序列独热编码
        left_padding_size = max(0, seq_len_half - dis_tx_start)
        right_padding_size = max(0, seq_len_half - dis_tx_end)
        seq_ref = 'N' * left_padding_size + seq[left_padding_size: seq_len - right_padding_size] + 'N' * right_padding_size
        x_ref = one_hot_encode(seq_ref)[None, :, :]
        # 方向为-，则反向互补
        if strands[i] == '-':
            x_ref = x_ref[:, ::-1, ::-1]
        # 剪切预测, 方向为-，则结果顺序反向
        y_ref = np.mean([model.predict(x_ref) for model in ann.models], axis=0)
        if strands[i] == '-':
            y_ref = y_ref[:, ::-1, :]
        # alt序列
        for alt in record.alts:
            len_alt = len(alt)
            len_del = max(len(record.ref) - len(alt), 0)
            # 跳过一些异常和无法预测的变异
            if '*' in alt or 'NON_REF' in alt or '.' in alt:
                LOGGER.warning(f'Skipping {record.chrom}-{record.pos}-{record.ref}-{alt}')
                continue
            # delins
            if len_ref > 1 and len_alt > 1:
                LOGGER.warning(f'Skipping {record.chrom}-{record.pos}-{record.ref}-{alt}')
                continue
            # 长del
            if len_del >= 2*distance:
                LOGGER.warning(f'Skipping {record.chrom}-{record.pos}-{record.ref}-{alt}')
                continue
            # 突变序列
            seq_alt = seq_ref[:seq_len_half] + str(alt) + seq_ref[seq_len_half + len_ref:]
            x_alt = one_hot_encode(seq_alt)[None, :, :]
            if strands[i] == '-':
                x_alt = x_alt[:, ::-1, ::-1]
            y_alt = np.mean([model.predict(x_alt) for model in ann.models], axis=0)
            if strands[i] == '-':
                y_alt = y_alt[:, ::-1, :]
            # del
            if len_ref > 1 and len_alt == 1:
                y_alt = np.concatenate([y_alt[:, :distance + len_alt, :],
                                        np.zeros((1, len_del, 3)),
                                        y_alt[:, distance + len_alt:, :]], axis=1)
            # ins
            elif len_ref == 1 and len_alt > 1:
                y_alt = np.concatenate([y_alt[:, :distance, :],
                                        np.max(y_alt[:, distance: distance + len_alt, :], axis=1)[:, None, :],
                                        y_alt[:, distance + len_alt:, :]], axis=1)
            # 得分差异最大的位置
            idx_gain_acceptor = (y_alt[0, :, 1] - y_ref[0, :, 1]).argmax()
            idx_loss_acceptor = (y_ref[0, :, 1] - y_alt[0, :, 1]).argmax()
            idx_gain_donor = (y_alt[0, :, 2] - y_ref[0, :, 2]).argmax()
            idx_loss_donor = (y_ref[0, :, 2] - y_alt[0, :, 2]).argmax()
            # 得分差异最大的位置是不是剪切位点决定进行屏蔽
            mask_gain_acceptor = np.logical_and((idx_gain_acceptor - distance) in dis_exon, is_mask)
            mask_loss_acceptor = np.logical_and((idx_loss_acceptor - distance) not in dis_exon, is_mask)
            mask_gain_donor = np.logical_and((idx_gain_donor - distance) in dis_exon, is_mask)
            mask_loss_donor = np.logical_and((idx_loss_donor - distance) not in dis_exon, is_mask)
            # 最大得分
            score_gain_acceptor = (y_alt[0, idx_gain_acceptor, 1] - y_ref[0, idx_gain_acceptor, 1]) * (1 - mask_gain_acceptor)
            score_loss_acceptor = (y_ref[0, idx_loss_acceptor, 1] - y_alt[0, idx_loss_acceptor, 1]) * (1 - mask_loss_acceptor)
            score_gain_donor = (y_alt[0, idx_gain_donor, 2] - y_ref[0, idx_gain_donor, 2]) * (1 - mask_gain_donor)
            score_loss_donor = (y_ref[0, idx_loss_donor, 2] - y_alt[0, idx_loss_donor, 2]) * (1 - mask_loss_donor)
            delta_scores.append(f'{alt}|{genes[i]}|{score_gain_acceptor:.2f}|{score_loss_acceptor:.2f}|{score_gain_donor:.2f}|{score_loss_donor:.2f}|{idx_gain_acceptor-distance}|{idx_loss_acceptor-distance}|{idx_gain_donor-distance}|{idx_loss_donor-distance}')
    return delta_scores
