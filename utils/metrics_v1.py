import numpy as np

precision_warn = "no positive predicted"
recall_warn = "no positive in ground truth"
f1_score_warn = "no TP or no positive in gt or predicted"
iou_warn = "no positive in both gt and predicted"

class Metrics:
    @staticmethod
    def computeMetrics(seg, gt, warn=True):
        assert seg.shape == gt.shape
        seg = seg.astype(np.uint8)
        gt = gt.astype(np.uint8)
        assert np.max(seg) <= 1
        assert np.max(gt) <= 1
        total = np.sum(np.ones_like(seg).astype(np.uint8))
        # 将total数据结构换成和TP FP FN 这些一样的int
        # 不能直接用int() numpy数据类型不同
        TP = np.sum(seg & gt)
        FP = np.sum(seg) - TP  # Overseg
        FN = np.sum(gt) - TP  # Underseg
        TN = total - TP - FP - FN

        accuracy = (TP + TN) / total  # 总样本里面预测对了多少个
        under_seg = FN / total
        over_seg = FP / total

        if (TP + FP) == 0:
            if warn:
                precision = precision_warn
            else:
                precision = 1
        else:
            precision = TP / (TP + FP)  # 预测出是正的里面有多少真正是正的

        if (TP + FN) == 0:
            if warn:
                recall = recall_warn
            else:
                recall = 1
        else:
            recall = TP / (TP + FN)  # 实际正样本中，分类器能预测出多少

        if (TP + FN) == 0 or (TP + FP) == 0 or TP == 0:
            if warn:
                f1_score = f1_score_warn
            else:
                f1_score = 1
        else:
            f1_score = 2 * recall * precision / (recall + precision)

        if (TP + FP + FN) == 0:
            if warn:
                iou = iou_warn
            else:
                iou = 1
        else:
            iou = TP / (TP + FP + FN)

        picScore = {}

        picScore['accuracy'] = accuracy
        picScore['under_seg'] = under_seg
        picScore['over_seg'] = over_seg
        picScore['precision'] = precision
        picScore['recall'] = recall
        picScore['f1_score'] = f1_score
        picScore['iou'] = iou

        picScore['TP'] = TP
        picScore['FP'] = FP
        picScore['FN'] = FN
        picScore['TN'] = TN
        picScore['Total'] = total

        return picScore
