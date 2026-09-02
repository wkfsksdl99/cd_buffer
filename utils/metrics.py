import numpy as np


def ap_per_class(
    true_positives,
    confidence,
    predicted_classes,
    target_classes,
    plot=False,
    save_dir='.',
    names=(),
):
    order = np.argsort(-confidence)
    true_positives = true_positives[order]
    confidence = confidence[order]
    predicted_classes = predicted_classes[order]

    unique_classes = np.unique(target_classes)
    x_axis = np.linspace(0, 1, 1000)
    average_precision = np.zeros(
        (len(unique_classes), true_positives.shape[1])
    )
    precision_curves = np.zeros((len(unique_classes), len(x_axis)))
    recall_curves = np.zeros((len(unique_classes), len(x_axis)))

    for class_index, class_id in enumerate(unique_classes):
        predicted_mask = predicted_classes == class_id
        label_count = (target_classes == class_id).sum()
        prediction_count = predicted_mask.sum()
        if prediction_count == 0 or label_count == 0:
            continue

        false_positives = (1 - true_positives[predicted_mask]).cumsum(0)
        accumulated_true_positives = true_positives[predicted_mask].cumsum(0)
        recall = accumulated_true_positives / (label_count + 1e-16)
        precision = accumulated_true_positives / (
            accumulated_true_positives + false_positives
        )
        recall_curves[class_index] = np.interp(
            -x_axis, -confidence[predicted_mask], recall[:, 0], left=0
        )
        precision_curves[class_index] = np.interp(
            -x_axis, -confidence[predicted_mask], precision[:, 0], left=1
        )
        for iou_index in range(true_positives.shape[1]):
            average_precision[class_index, iou_index], _, _ = compute_ap(
                recall[:, iou_index], precision[:, iou_index]
            )

    f1_curves = (
        2
        * precision_curves
        * recall_curves
        / (precision_curves + recall_curves + 1e-16)
    )
    best_index = f1_curves.mean(0).argmax()
    return (
        precision_curves[:, best_index],
        recall_curves[:, best_index],
        average_precision,
        f1_curves[:, best_index],
        unique_classes.astype('int32'),
    )


def compute_ap(recall, precision):
    recall_curve = np.concatenate(([0.0], recall, [recall[-1] + 0.01]))
    precision_curve = np.concatenate(([1.0], precision, [0.0]))
    precision_curve = np.flip(
        np.maximum.accumulate(np.flip(precision_curve))
    )
    x_axis = np.linspace(0, 1, 101)
    average_precision = np.trapz(
        np.interp(x_axis, recall_curve, precision_curve), x_axis
    )
    return average_precision, precision_curve, recall_curve
