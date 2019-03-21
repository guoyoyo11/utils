#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Information Retrieval metrics
Useful Resources:
http://www.cs.utexas.edu/~mooney/ir-course/slides/Evaluation.ppt
http://www.nii.ac.jp/TechReports/05-014E.pdf
http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
http://hal.archives-ouvertes.fr/docs/00/72/67/60/PDF/07-busa-fekete.pdf
"""

import numpy as np
import sys
import random


def mean_reciprocal_rank(rs):
    """Score is reciprocal of the rank of the first relevant item
    First element is 'rank 1'.  Relevance is binary (nonzero is relevant).
    Example from http://en.wikipedia.org/wiki/Mean_reciprocal_rank
    >>> rs = [[0, 0, 1], [0, 1, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.61111111111111105
    >>> rs = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])
    >>> mean_reciprocal_rank(rs)
    0.5
    >>> rs = [[0, 0, 0, 1], [1, 0, 0], [1, 0, 0]]
    >>> mean_reciprocal_rank(rs)
    0.75
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean reciprocal rank
    """
    rs = (np.asarray(r).nonzero()[0] for r in rs)
    return np.mean([1. / (r[0] + 1) if r.size else 0. for r in rs])


def r_precision(r):
    """Score is precision after all relevant documents have been retrieved
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> r_precision(r)
    0.33333333333333331
    >>> r = [0, 1, 0]
    >>> r_precision(r)
    0.5
    >>> r = [1, 0, 0]
    >>> r_precision(r)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        R Precision
    """
    r = np.asarray(r) != 0
    z = r.nonzero()[0]
    if not z.size:
        return 0.
    return np.mean(r[:z[-1] + 1])


def recall_at_k(r, k):
    """Score is recall @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 1, 1, 1]
    >>> recall_at_k(r, 1)
    0.0
    >>> recall_at_k(r, 2)
    0.5
    >>> recall_at_k(r, 3)
    0.6666666666666666
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Recall @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    r = np.asarray(r) != 0
    assert k >= 1
    if r.size < k:
        raise ValueError('Relevance score length < k')
    z = r.nonzero()[0]
    rn = k if z.size >= k else z.size
    rk = float(z[z < k].size) / rn if rn else 0.0
    # print rk, z
    return rk


def precision_at_k(r, k):
    """Score is precision @ k
    Relevance is binary (nonzero is relevant).
    >>> r = [0, 0, 1]
    >>> precision_at_k(r, 1)
    0.0
    >>> precision_at_k(r, 2)
    0.0
    >>> precision_at_k(r, 3)
    0.33333333333333331
    >>> precision_at_k(r, 4)
    Traceback (most recent call last):
        File "<stdin>", line 1, in ?
    ValueError: Relevance score length < k
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Precision @ k
    Raises:
        ValueError: len(r) must be >= k
    """
    assert k >= 1
    r = np.asarray(r)[:k] != 0
    if r.size != k:
        raise ValueError('Relevance score length < k')
    return np.mean(r)


def average_precision(r):
    """Score is average precision (area under PR curve)
    Relevance is binary (nonzero is relevant).
    >>> r = [1, 1, 0, 1, 0, 1, 0, 0, 0, 1]
    >>> delta_r = 1. / sum(r)
    >>> sum([sum(r[:x + 1]) / (x + 1.) * delta_r for x, y in enumerate(r) if y])
    0.7833333333333333
    >>> average_precision(r)
    0.78333333333333333
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Average precision
    """
    # print r
    r = np.asarray(r) != 0
    out = [precision_at_k(r, k + 1) for k in range(r.size) if r[k]]
    ap = np.mean(out) if out else 0.
    # print "%d\t%.3f" %(k+1, ap)
    return ap


def mean_average_precision(rs):
    """Score is mean average precision
    Relevance is binary (nonzero is relevant).
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1]]
    >>> mean_average_precision(rs)
    0.78333333333333333
    >>> rs = [[1, 1, 0, 1, 0, 1, 0, 0, 0, 1], [0]]
    >>> mean_average_precision(rs)
    0.39166666666666666
    Args:
        rs: Iterator of relevance scores (list or numpy) in rank order
            (first element is the first item)
    Returns:
        Mean average precision
    """
    return np.mean([average_precision(r) for r in rs])


def dcg_at_k(r, k, method=0):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> dcg_at_k(r, 1)
    3.0
    >>> dcg_at_k(r, 1, method=1)
    3.0
    >>> dcg_at_k(r, 2)
    5.0
    >>> dcg_at_k(r, 2, method=1)
    4.2618595071429155
    >>> dcg_at_k(r, 10)
    9.6051177391888114
    >>> dcg_at_k(r, 11)
    9.6051177391888114
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=0):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Example from
    http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
    >>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
    >>> ndcg_at_k(r, 1)
    1.0
    >>> r = [2, 1, 2, 0]
    >>> ndcg_at_k(r, 4)
    0.9203032077642922
    >>> ndcg_at_k(r, 4, method=1)
    0.96519546960144276
    >>> ndcg_at_k([0], 1)
    0.0
    >>> ndcg_at_k([1], 2)
    1.0
    Args:
        r: Relevance scores (list or numpy) in rank order
            (first element is the first item)
        k: Number of results to consider
        method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
                If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def test():
    """ test
    """
    import doctest
    doctest.testmod()


def _load_evalfile(filename):
    """
    File: evaluate ranked responses
    """
    n = len(CS)
    eval_list, scores, columns = [[] for i in range(n)], [[] for i in range(n)], []
    top1_list = []
    with open(filename) as fi:
        for line in fi.readlines():
            if line.strip() == "": continue
            # rank \t query \t response \t score
            columns = line.strip().split('\t')
            assert len(columns) >= CS[n - 1]
            if columns[0] == "1":  # the first column is rank
                for i in range(n):
                    # there is at least one relevant response to be evaluated
                    if (scores[i] and (EVAL_ALL or max(scores[i]))): eval_list[i].append(scores[i])
                    scores[i] = []
                if n > 1:  # record the top1 for evaluation
                    top1 = [columns[1].strip("\"")]
                    if random.random() < 0.5:
                        for c in CS: top1.extend((columns[c - 2].strip("\""), columns[c - 1]))
                        top1.append(0)
                    else:  # reverse the two systems
                        for c in CS[::-1]: top1.extend((columns[c - 2].strip("\""), columns[c - 1]))
                        top1.append(1)
                    top1_list.append(top1)
            next = False
            for c in CS:
                if c - 2 >= 0 and columns[c - 2] in IGNORE:
                    next = True
                    break
            if next: continue
            for i in range(n):
                # -1 is only used for dcg
                scores[i].append(int(columns[CS[i] - 1]) if int(columns[CS[i] - 1]) >= 0 else 0)
        # the last query
        for i in range(n):
            if (scores[i] and (EVAL_ALL or max(scores[i]))): eval_list[i].append(scores[i])
    return eval_list, top1_list


def _print_list(filename, llist):
    with open(filename, 'w') as fp:
        for list in llist:
            for li in list:
                fp.write("%s\t" % li)
            fp.write("\n")


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print "usage:", sys.argv[0], "file_to_be_evaluated"
        exit(-1)
    # evaluate all queries (or queries with at least one relevant response)
    EVAL_ALL = (sys.argv[2] in ("true", "True", "T")) if len(sys.argv) > 2 else True
    CS = (4, 6)  # the column of score in the file: start from 1
    IGNORE = ("None")  # output to be ignored
    # read the file to be evaluated
    eval_list = _load_evalfile(sys.argv[1])
    pk, rk = 1, 5
    for i in range(len(CS)):
        print "system_%d:" % (i + 1)
        print "Totally %d queries and %d responses" % \
              (len(eval_list[i]), np.sum(len(r) for r in eval_list[i]))
        print "Precision@%d = %.3f" % \
              (pk, np.mean([precision_at_k(r, pk) for r in eval_list[i]]))
        print "Recall@%d = %.3f" % (rk, np.mean([recall_at_k(r, rk) for r in eval_list[i]]))
        print "R_Precision = %.3f" % np.mean([r_precision(r) for r in eval_list[i]])
        print "MAP = %.3f" % mean_average_precision(eval_list[i])
        print "MRR = %.3f" % mean_reciprocal_rank(eval_list[i])
