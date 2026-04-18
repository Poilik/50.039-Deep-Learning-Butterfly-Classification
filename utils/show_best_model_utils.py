def show_best_model(model, ckpt_file):
    hist = model.load_checkpoint_history(ckpt_file)

    valid = [
    r for r in hist
    if r.get("metrics", {}).get("val_f2_per_class_history")
    and len(r["metrics"]["val_f2_per_class_history"]) > 0
    ]

    best = max(
    valid,
    key=lambda r: float(r["metrics"]["val_f2_per_class_history"][-1][1])
    )

    print("Selected epoch:", best["epoch"])
    print("Best val F2(class 0):", float(best["metrics"]["val_f2_per_class_history"][-1][0]))
    print("Best val F2(class 1):", float(best["metrics"]["val_f2_per_class_history"][-1][1]))

    return best, best["metrics"]["val_f2_per_class_history"][-1][1]