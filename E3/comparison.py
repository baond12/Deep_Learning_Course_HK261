"""Compare current RNNs with explicitly sourced, fixed E1 baseline results."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common.plots import plot_learning_curves, save_summary_csv
from E3.models import TOKENIZATIONS


def write_comparisons(experiments: dict, histories: dict, out_dir: Path,
                      baselines: dict, baseline_histories: dict) -> None:
    pairs, probes = [], []
    for kind in TOKENIZATIONS:
        a,b = experiments.get(f"lstm_{kind}"),experiments.get(f"gru_{kind}")
        row = {"tokenization":kind,"pair_complete":a is not None and b is not None}
        if a is not None and b is not None:
            row.update(lstm_test_accuracy=a["test_accuracy"],gru_test_accuracy=b["test_accuracy"],
                       gru_minus_lstm_accuracy_pp=100*(b["test_accuracy"]-a["test_accuracy"]),
                       lstm_parameters=a["parameters"],gru_parameters=b["parameters"],
                       lstm_train_val_seconds=a["training_time_seconds"],gru_train_val_seconds=b["training_time_seconds"])
        pairs.append(row)
        group = {k:v for k,v in histories.items() if k in (f"lstm_{kind}",f"gru_{kind}")}
        if group:
            plot_learning_curves(group,out_dir/f"learning_curves_{kind}.png")
    keys = list(dict.fromkeys(k for r in pairs for k in r))
    save_summary_csv([{k:r.get(k) for k in keys} for r in pairs],out_dir/"comparison_rnn.csv")
    combined = {**experiments, **baselines}
    fields = ("experiment","family","sequence","sequence_length","feature_dim","test_accuracy","macro_f1","test_loss",
              "parameters","training_time_seconds","source","trained_in_this_run","epochs","batch_size",
              "source_metrics","source_metrics_sha256")
    save_summary_csv([{k:r.get(k) for k in fields} for r in combined.values()],out_dir/"comparison_baselines.csv")
    for name,result in experiments.items():
        probe = result.get("order_probe")
        if probe:
            normal = probe["orders"]["raster"]["accuracy"]
            for order,score in probe["orders"].items():
                probes.append({"experiment":name,"split":"validation","order":order,"accuracy":score["accuracy"],
                               "loss":score["loss"],"drop_from_raster_pp":100*(normal-score["accuracy"])})
    if probes:
        save_summary_csv(probes,out_dir/"comparison_order_probe.csv")
    plot_learning_curves({f"E1_{k}": v for k, v in baseline_histories.items()},out_dir/"learning_curves_baselines_e1.png")
    names = list(combined)
    labels = [f"E1: {n}" if n in baselines else f"E3: {n}" for n in names]
    fig,axes = plt.subplots(1,3,figsize=(17,6))
    for ax,key,label,scale in zip(axes,("test_accuracy","parameters","training_time_seconds"),
                                ("Test accuracy (%)","Trainable parameters","Train + validation (s)"),(100,1,1)):
        ax.barh(labels,[combined[n][key]*scale for n in names],color="0.35")
        ax.set_xlabel(label)
        ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir/"comparison.png",dpi=160)
    plt.close(fig)
    text = ["# E3 — Kết quả của lượt chạy hiện tại","",
            "Huấn luyện tám LSTM/GRU ở E3; MLP/CNN lấy nguyên kết quả E1 đã có, không train hoặc test lại.",
            "Baseline được kiểm tra cùng split và cấu hình epochs/batch/lr/weight decay; nguồn và SHA-256 nằm trong baselines_e1.json.",""]
    for baseline in baselines.values():
        text.append(f"- E1 {baseline['family']}: test accuracy {100*baseline['test_accuracy']:.2f}%; nguồn `{baseline['source_metrics']}`.")
    for row in pairs:
        text.append(f"- {row['tokenization']}: GRU trừ LSTM = {row['gru_minus_lstm_accuracy_pp']:+.2f} điểm phần trăm accuracy test."
                    if row["pair_complete"] else f"- {row['tokenization']}: chưa đủ cặp LSTM/GRU.")
    text += ["","Một seed chưa đủ kết luận độ ổn định. Tokenizer khác nhau còn thay đổi chiều đặc trưng và capacity; CNN-stem bổ sung CNN.",
             "Ảnh là chuỗi vị trí không gian, không phải thời gian thực. Order probe chỉ đo độ nhạy thứ tự trên validation với checkpoint cố định, không chứng minh trí nhớ dài hạn.",
             "E1 và E3 khác lượt chạy và cách tiêu thụ RNG; E1 thiếu metadata GPU. Thời gian E1 là số đo lịch sử, không đủ để xếp hạng tốc độ trực tiếp với E3.",
             "Thời gian train + validation không phải latency suy luận; tải GPU cạnh tranh có thể ảnh hưởng."]
    (out_dir/"comparison.md").write_text("\n".join(text)+"\n",encoding="utf-8")
