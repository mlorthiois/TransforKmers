from typing import Tuple, Dict
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from .postprocess import write_in_csv

log = logging.getLogger(__name__)
plt.style.use("seaborn-deep")


def evaluation_prediction_distribution(
    y, y_pred_softmax, mask, threshold, ids, output_prefix=None, header=""
) -> None:
    filename = f"{output_prefix}prediction_distribution"
    write_in_csv(f"{filename}.csv", (ids, *y, y_pred_softmax), header=header)

    # distribution plot
    fig, ax = plt.subplots()
    ax.axvspan(0, threshold, color="#6AA66E", alpha=0.25)
    ax.axvspan(threshold, 1, color="#5471AB", alpha=0.25)
    ax.hist(
        (y_pred_softmax[mask], y_pred_softmax[~mask]),
        label=("True genes", "False genes"),
        bins=50,
        stacked=True,
        edgecolor="black",
        linewidth=0.5,
    )
    ax.axvline(x=threshold, ls="--", color="black", linewidth=1)
    ax.set(
        title="Distributions of predictions",
        xlabel="Positive probability (predicted)",
        ylabel="Samples",
    )
    ax.margins(x=0)
    ax.legend(loc="upper left")
    fig.savefig(f"{filename}.pdf", format="pdf")
    log.info(f"{filename}.pdf created.")
    return


def confusion_matrix(y_true, y_pred_labels, output_prefix=None) -> None:
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred_labels).ravel()
    write_in_csv(
        f"{output_prefix}confusion_matrix.csv",
        ([tn], [fp], [fn], [tp]),
        header="true_negative,false_positive,false_negative,true_positive",
    )

    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=np.array([[tn, fp], [fn, tp]]))
    disp.plot(cmap="Blues")
    plt.title("Confusion matrix")
    plt.savefig(f"{output_prefix}confusion_matrix.pdf", format="pdf")
    log.info(f"{output_prefix}confusion_matrix.pdf created")
    return


def classification_report(y_true, y_pred_labels, output_prefix=None) -> None:
    report = metrics.classification_report(y_true, y_pred_labels, output_dict=True)
    del report["accuracy"]
    report_metrics = ["precision", "recall", "f1-score"]
    labels = {
        "0": "False genes",
        "1": "True genes",
        "macro avg": "Macro avg",
        "weighted avg": "Weighted avg",
    }

    with open(f"{output_prefix}classification_report.csv", "w") as fd:
        fd.write(f"label,{','.join(report_metrics)},support\n")
        for label, values in report.items():
            label_name = labels[label]
            rendered_label_metrics = ",".join(map(str, list(values.values())))
            fd.write(f"{label_name},{rendered_label_metrics}\n")
    log.info(f"{output_prefix}classification_report.csv created")

    # Histogram
    width = 0.75
    x = np.arange(len(labels))
    fig, ax = plt.subplots()
    for i, metric in enumerate(report_metrics):
        if metric == "support":
            continue
        ax.bar(
            x + (i - 1) * width / len(report_metrics),  # x - width / 3, x, x - width / 3
            [report[v][metric] for v in labels],
            width / len(report_metrics),
            label=metric,
            linewidth=1,
            edgecolor="black",
        )
    ax.set(title="Classification report", ylabel="Scores", ylim=(0.7, 1))
    ax.set_xticks(x, labels=list(labels.values()))
    leg = ax.legend(loc="upper right")
    leg.get_frame().set(alpha=None, edgecolor="black", boxstyle="square")
    fig.savefig(f"{output_prefix}classification_report.pdf", format="pdf")
    log.info(f"{output_prefix}classification_report.pdf created")
    return


def curve_plot(
    axes,
    filename,
    show_th=True,
    threshold=None,
    annotext="",
    point_coord=(0.5, 0.5),
    line=None,
    xlabel="x",
    ylabel="y",
    title="Plot",
):
    fig, ax = plt.subplots()

    for x, y, label in axes:
        ax.plot(x, y, marker=".", label=label)
    for l in ax.lines:
        l.set_marker(None)

    if line is not None:
        ax.plot(*line, "k--", label="No Skill")
    if show_th and threshold is not None:
        ax.scatter(*threshold, color="red", marker="o", label="Optimal")
        ax.annotate(
            annotext,
            xy=threshold,
            xytext=point_coord,
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
        )
    ax.legend()
    ax.margins(x=0)
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title, xlim=(0, 1))
    fig.savefig(filename, format="pdf")
    log.info(f"{filename} created")
    return


def roc_curve(y_true, y_pred_softmax, output_prefix, **kwargs) -> Tuple[np.array, np.array, int]:
    fpr, tpr, threshold = metrics.roc_curve(y_true, y_pred_softmax)
    write_in_csv(f"{output_prefix}roc_curve.csv", (fpr, tpr, threshold), header="fpr,tpr,threshold")
    th_idx = np.argmax(tpr - fpr)
    curve_plot(
        [(fpr, tpr, "metric_label")],
        f"{output_prefix}roc_curve.pdf",
        threshold=(fpr[th_idx], tpr[th_idx]),
        point_coord=(0.25, 0.65),
        annotext=f"threshold={threshold[th_idx]:.3f}",
        title="ROC",
        line=([0, 1], [0, 1]),
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        **kwargs,
    )
    return tpr - fpr, threshold, th_idx


def precision_recall_curve(
    y_true, y_pred_softmax, output_prefix, **kwargs
) -> Tuple[np.array, np.array, int]:
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_pred_softmax)
    write_in_csv(
        f"{output_prefix}precision_recall_curve.csv",
        (precision[:-1], recall[:-1], thresholds),
        header="precision,recall,threshold",
    )
    f1score = (2 * precision * recall) / (precision + recall)
    f1_idx = np.argmax(f1score)
    curve_plot(
        [(recall, precision, None)],
        f"{output_prefix}precision_recall_curve.pdf",
        threshold=(recall[f1_idx], precision[f1_idx]),
        annotext=f"threshold={thresholds[f1_idx]:.3f}",
        point_coord=(0.5, 0.85),
        title="Precision-recall curve",
        line=([0, 1], [0.5, 0.5]),
        xlabel="Precision",
        ylabel="Recall",
        **kwargs,
    )
    return f1score, thresholds, f1_idx


def training_history(file, output_prefix="./") -> None:
    import json

    log.info("Parsing training history...")
    with open(file) as fd:
        history = json.load(fd)["log_history"]

    metrics = {
        "train": {"epochs": [], "loss": [], "learning_rate": []},
        "eval": {"epochs": [], "loss": [], "learning_rate": []},
    }

    for step in history:
        epoch = step["epoch"]

        if "eval_loss" in step:
            mode = "eval"
            loss = step["eval_loss"]
            lr = np.nan
        else:
            mode = "train"
            loss = step["loss"]
            lr = step["learning_rate"]

        metrics[mode]["epochs"].append(epoch)
        metrics[mode]["loss"].append(loss)
        metrics[mode]["learning_rate"].append(lr)

    write_in_csv(
        f"{output_prefix}training_history.csv",
        (
            ["train"] * len(metrics["train"]["epochs"]) + ["eval"] * len(metrics["eval"]["epochs"]),
            metrics["train"]["loss"] + metrics["eval"]["loss"],
            metrics["train"]["epochs"] + metrics["eval"]["epochs"],
            metrics["train"]["learning_rate"] + metrics["eval"]["learning_rate"],
        ),
        header=["mode", "loss", "epoch", "learning_rate"],
    )

    plt.style.use("seaborn-deep")
    fig, ax = plt.subplots()
    ax.plot(metrics["train"]["epochs"], metrics["train"]["loss"], label="train loss")
    ax.plot(metrics["eval"]["epochs"], metrics["eval"]["loss"], label="eval loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")

    ax2 = ax.twinx()
    color = "tab:red"
    ax2.set_ylabel("Learning rate", color=color)  # we already handled the x-label with ax1
    ax2.plot(metrics["train"]["epochs"], metrics["train"]["learning_rate"], color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    ax.legend()
    ax.set_title("Training and evaluation losses")
    fig.tight_layout()
    fig.savefig(f"{output_prefix}training_loss.pdf")
    log.info(f"{output_prefix}training_loss.pdf created.")
    return


def accuracy_curve(y, y_pred_softmax, output_prefix):
    acc_th = np.arange(0, 1.05, 0.005)
    acc = np.array(
        [metrics.accuracy_score(y, np.where(y_pred_softmax < th, 0, 1)) for th in acc_th]
    )
    write_in_csv(
        f"{output_prefix}accuracy_curve.csv",
        (acc_th, acc),
        header="threshold,accuracy",
    )
    return acc, acc_th, acc.argmax(-1)


def find_best_threshold(y, y_pred_softmax, output_prefix) -> Dict[str, float]:
    roc, roc_th, roc_id = roc_curve(y, y_pred_softmax, output_prefix=output_prefix)
    pr, pr_th, pr_id = precision_recall_curve(y, y_pred_softmax, output_prefix=output_prefix)
    acc, acc_th, acc_id = accuracy_curve(y, y_pred_softmax, output_prefix=output_prefix)

    curve_plot(
        [(roc_th, roc, "TPR-FPR"), (pr_th, pr[:-1], "F1 score"), (acc_th, acc, "Accuracy")],
        f"{output_prefix}/threshold.pdf",
        title="Classification threshold",
        xlabel="Threshold",
        ylabel="Metric",
    )

    thresholds = {
        "roc": roc_th[roc_id].item(),
        "pr": pr_th[pr_id].item(),
        "accuracy": acc_th[acc_id].item(),
    }
    thresholds["mean"] = sum(thresholds.values()) / len(thresholds)

    with open(f"{output_prefix}/threshold_best.csv", "w") as fd:
        fd.write("metric,best_threshold\n")
        for metric, value in thresholds.items():
            fd.write(f"{metric},{value}\n")

    return thresholds
