# IoU
def IoU(box1, box2):
    #        x1  y1  x2  y2
    # box = [ 0,  1,  2,  3]
    area1 = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    area2 = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    inter = (min(box1[2], box2[2]) - max(box1[0], box2[0]) + 1) * (min(box1[3], box2[3]) - max(box1[1], box2[1]) + 1)
    union = area1 + area2 - inter

    return inter / union
