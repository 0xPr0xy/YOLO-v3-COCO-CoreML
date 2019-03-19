import Foundation
import UIKit
import CoreML
import Accelerate

// The labels for the 80 classes.
let labels = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hotdog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "plant",
    "bed",
    "table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush"
]

// anchor boxes
let anchors: [Float] = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]

/**
 Removes bounding boxes that overlap too much with other boxes that have
 a higher score.

 Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc

 - Parameters:
 - boxes: an array of bounding boxes and their scores
 - limit: the maximum number of boxes that will be selected
 - threshold: used to decide whether boxes overlap too much
 */
func nonMaxSuppression(boxes: [YOLO.Prediction], limit: Int, threshold: Float) -> [YOLO.Prediction] {

    // Do an argsort on the confidence scores, from high to low.
    let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }

    var selected: [YOLO.Prediction] = []
    var active = [Bool](repeating: true, count: boxes.count)
    var numActive = active.count

    // The algorithm is simple: Start with the box that has the highest score.
    // Remove any remaining boxes that overlap it more than the given threshold
    // amount. If there are any boxes left (i.e. these did not overlap with any
    // previous boxes), then repeat this procedure, until no more boxes remain
    // or the limit has been reached.
    outer: for i in 0..<boxes.count {
        if active[i] {
            let boxA = boxes[sortedIndices[i]]
            selected.append(boxA)
            if selected.count >= limit { break }

            for j in i+1..<boxes.count {
                if active[j] {
                    let boxB = boxes[sortedIndices[j]]
                    if IOU(a: boxA.rect, b: boxB.rect) > threshold {
                        active[j] = false
                        numActive -= 1
                        if numActive <= 0 { break outer }
                    }
                }
            }
        }
    }
    return selected
}

/**
 Computes intersection-over-union overlap between two bounding boxes.
 */
public func IOU(a: CGRect, b: CGRect) -> Float {
    let areaA = a.width * a.height
    if areaA <= 0 { return 0 }

    let areaB = b.width * b.height
    if areaB <= 0 { return 0 }

    let intersectionMinX = max(a.minX, b.minX)
    let intersectionMinY = max(a.minY, b.minY)
    let intersectionMaxX = min(a.maxX, b.maxX)
    let intersectionMaxY = min(a.maxY, b.maxY)
    let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
        max(intersectionMaxX - intersectionMinX, 0)
    return Float(intersectionArea / (areaA + areaB - intersectionArea))
}

extension Array where Element: Comparable {
    /**
     Returns the index and value of the largest element in the array.
     */
    public func argmax() -> (Int, Element) {
        precondition(self.count > 0)
        var maxIndex = 0
        var maxValue = self[0]
        for i in 1..<self.count {
            if self[i] > maxValue {
                maxValue = self[i]
                maxIndex = i
            }
        }
        return (maxIndex, maxValue)
    }
}

/**
 Logistic sigmoid.
 */
public func sigmoid(_ x: Float) -> Float {
    return 1 / (1 + exp(-x))
}

/**
 Computes the "softmax" function over an array.

 Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/

 This is what softmax looks like in "pseudocode" (actually using Python
 and numpy):

 x -= np.max(x)
 exp_scores = np.exp(x)
 softmax = exp_scores / np.sum(exp_scores)

 First we shift the values of x so that the highest value in the array is 0.
 This ensures numerical stability with the exponents, so they don't blow up.
 */
public func softmax(_ x: [Float]) -> [Float] {
    var x = x
    let len = vDSP_Length(x.count)

    // Find the maximum value in the input array.
    var max: Float = 0
    vDSP_maxv(x, 1, &max, len)

    // Subtract the maximum from all the elements in the array.
    // Now the highest value in the array is 0.
    max = -max
    vDSP_vsadd(x, 1, &max, &x, 1, len)

    // Exponentiate all the elements in the array.
    var count = Int32(x.count)
    vvexpf(&x, x, &count)

    // Compute the sum of all exponentiated values.
    var sum: Float = 0
    vDSP_sve(x, 1, &sum, len)

    // Divide each element by the sum. This normalizes the array contents
    // so that they all add up to 1.
    vDSP_vsdiv(x, 1, &sum, &x, 1, len)

    return x
}
