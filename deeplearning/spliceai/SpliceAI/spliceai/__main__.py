from argparse import ArgumentParser
from multiprocessing import Process, cpu_count, Queue
import queue
from collections import namedtuple
from itertools import count
from pysam import VariantFile
from spliceai.utils import get_delta_scores, Annotator


# 多进程序列化变异位点
Variant = namedtuple('Variant', ['id', 'chrom', 'pos', 'ref', 'alts'])


def process_record(records: Queue, results: Queue, ref_fasta: str, annotations: str, distance: int, is_mask: int):
    # 工作进程负责从输入队列获取变异，预测完塞进结果队列。输入队列遇到终止信号后退出
    # 预测实例
    ann = Annotator(ref_fasta, annotations)
    # 监听队列
    while True:
        # 输入队列中获取变异
        try:
            record = records.get_nowait()
        except queue.Empty:
            continue
        # 变异信息非终止信号，则进行预测
        if record != 'END':
            scores = get_delta_scores(record, ann, distance, is_mask)
            # 预测结果写入队列，id与变异信息对应
            results.put((record.id, scores))
        else:
            # 为其他进程留一个结束信号
            records.put('END')
            break


def run_parallel(args):
    # 多进程数
    cpu_num = min(cpu_count(), args.P)
    # 创建输入和输出队列
    records, results = Queue(10 * cpu_num), Queue()
    # 创建多进程进行预测
    for _ in range(cpu_num):
        p = Process(target=process_record, args=(records, results, args.R, args.A, args.D, args.M))
        p.start()
    # 读取vcf，输出vcf
    vcf = VariantFile(args.I)
    header = vcf.header
    header.add_line('##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAIv1.3.1 variant '
                'annotation. These include delta scores (DS) and delta positions (DP) for '
                'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
                'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">')
    out = VariantFile(args.O, 'w', header=vcf.header)
    # 读取vcf信息的初始状态, 写入新的vcf的初始状态，原始变异信息存储字典，变异id
    input_finished = False
    output_finished = False
    wait_records = dict()
    records_id = count()
    while True:
        # 输入未读完成，且输入队列未满时循环去获取变异信息
        while not input_finished and not records.full():
            try:
                # 变异关键信息塞入队列，需要能序列化才能使用多进程
                record_id, record = next(records_id), next(vcf)
                records.put(Variant(record_id, record.chrom, record.pos, record.ref, record.alts))
                # 变异全部信息存储待写入新的vcf
                wait_records[record_id] = record
            except StopIteration:
                # 输入读取完成，则写入队列终止信号，flag置为True
                records.put('END')
                input_finished = True
        # 变异全部信息存储字典非空时循环
        while wait_records:
            # 结果队列获取预测结果，队列为空的时候退出循环
            try:
                record_id, scores = results.get_nowait()
            except queue.Empty:
                break
            # 根据id从变异信息存储字典中移除
            _record = wait_records.pop(record_id)
            # 预测有值增加SpliceAI信息
            if len(scores) > 0:
                _record.info['SpliceAI'] = scores
            out.write(_record)
        else:
            # 变异信息全部重新输出完毕flag置为True
            output_finished = True
        # 全部处理完成退出主进程循环
        if output_finished:
            break
    # 关闭文件
    vcf.close()
    out.close()


def run_serial(args):
    ann = Annotator(args.R, args.A)
    vcf = VariantFile(args.I)
    header = vcf.header
    header.add_line('##INFO=<ID=SpliceAI,Number=.,Type=String,Description="SpliceAIv1.3.1 variant '
                    'annotation. These include delta scores (DS) and delta positions (DP) for '
                    'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL). '
                    'Format: ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">')
    out = VariantFile(args.O, 'w', header=header)
    for record in vcf:
        scores = get_delta_scores(record, ann, args.D, args.M)
        if len(scores) > 0:
            record.info['SpliceAI'] = scores
        out.write(record)
    vcf.close()
    out.close()


def main():
    parser = ArgumentParser()
    parser.add_argument('-A', help='Genome hg19 or hg38', default='grch37', type=str)
    parser.add_argument('-I', help='Input file', required=True)
    parser.add_argument('-O', help='Output file', required=True)
    parser.add_argument('-M', help='mask not splice site', choices=[0, 1], type=int, default=0)
    parser.add_argument('-D', help='distance', type=int, default=50)
    parser.add_argument('-R', help='reference fasta file', required=True)
    parser.add_argument('-P', help='process number', type=int, default=1)
    args = parser.parse_args()
    run_serial(args) if min(args.P, cpu_count()) == 1 else run_parallel(args)


if __name__ == '__main__':
    main()
