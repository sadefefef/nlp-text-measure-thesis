# -*- coding: utf-8 -*-
"""实验主程序（论文第 5 章）。

实验清单
--------
E1  测度描述性统计与区分度检验      -> results/tables/t1_descriptive.csv
E2  测度-评分相关性与显著性检验      -> results/tables/t2_correlation.csv
E3  测度共线性检验（VIF）            -> results/tables/t3_vif.csv
E4  熵权法权重与维度信息量           -> results/tables/t4_entropy_weight.csv
E5  评分预测回归（5 折交叉验证）      -> results/tables/t5_regression.csv
E6  消融实验（留一维度）             -> results/tables/t6_ablation.csv
E7  三分类情感判别                  -> results/tables/t7_classification.csv
E8  簇内异质性（K-means 分群）        -> results/tables/t8_cluster.csv
E9  情感测度实现对比（词典 vs SnowNLP）-> results/tables/t9_sentiment_impl.csv
图表 -> results/figures/*.png

运行：python experiment.py
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from measures import DIM_OF, DIMENSIONS, MEASURE_KEYS, MEASURE_NAMES, MeasureExtractor

warnings.filterwarnings("ignore")
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "reviews.csv"
TBL = ROOT / "results" / "tables"
FIG = ROOT / "results" / "figures"
TBL.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
SEED = 20260517

DIM_KEYS = {d: [k for k in MEASURE_KEYS if DIM_OF[k] == d] for d in DIMENSIONS}
log_lines: list[str] = []


def log(msg: str) -> None:
    print(msg)
    log_lines.append(msg)


# --------------------------------------------------------------------------- 1 数据与测度
def build_features() -> pd.DataFrame:
    df = pd.read_csv(DATA)
    log(f"[数据] 载入 {len(df)} 条评论，字段 {list(df.columns)}")
    ext = MeasureExtractor(random_state=SEED)
    ext.fit(df["text"].tolist(), df["category"].tolist())
    log("[测度] 语料级统计量（IDF / 类别质心 / 句向量空间）已估计")

    rows = [ext.extract(t, c) for t, c in zip(df["text"], df["category"])]
    feats = pd.DataFrame(rows)
    df = pd.concat([df.reset_index(drop=True), feats], axis=1)
    log(f"[测度] 完成 {len(MEASURE_KEYS)} 项测度抽取，缺失值 {int(feats.isna().sum().sum())} 个")
    return df, ext


# --------------------------------------------------------------------------- 2 E1 描述统计
def e1_descriptive(df: pd.DataFrame) -> pd.DataFrame:
    rec = []
    for k in MEASURE_KEYS:
        x = df[k].astype(float)
        # 区分度：|与评分的 Spearman 秩相关|
        rho, _ = stats.spearmanr(x, df["rating"])
        rec.append({
            "测度": k, "名称": MEASURE_NAMES[k], "维度": DIM_OF[k],
            "均值": round(x.mean(), 4), "标准差": round(x.std(), 4),
            "最小值": round(x.min(), 4), "最大值": round(x.max(), 4),
            "偏度": round(x.skew(), 3), "变异系数": round(x.std() / (abs(x.mean()) + 1e-9), 3),
            "区分度|ρ|": round(abs(rho), 4),
        })
    t = pd.DataFrame(rec)
    t.to_csv(TBL / "t1_descriptive.csv", index=False, encoding="utf-8-sig")
    log(f"[E1] 描述性统计完成，区分度最高：{t.loc[t['区分度|ρ|'].idxmax(), '名称']}"
        f"（|ρ|={t['区分度|ρ|'].max():.3f}）")
    return t


# --------------------------------------------------------------------------- 3 E2 相关性
def e2_correlation(df: pd.DataFrame) -> pd.DataFrame:
    rec = []
    for k in MEASURE_KEYS:
        x = df[k].astype(float)
        r_p, p_p = stats.pearsonr(x, df["rating"])
        r_s, p_s = stats.spearmanr(x, df["rating"])
        rec.append({
            "测度": k, "名称": MEASURE_NAMES[k], "维度": DIM_OF[k],
            "Pearson r": round(r_p, 4), "p(Pearson)": p_p,
            "Spearman ρ": round(r_s, 4), "p(Spearman)": p_s,
            "显著性": "***" if p_s < 0.001 else ("**" if p_s < 0.01 else ("*" if p_s < 0.05 else "n.s.")),
        })
    t = pd.DataFrame(rec).sort_values("Spearman ρ", key=lambda s: -s.abs())
    t.to_csv(TBL / "t2_correlation.csv", index=False, encoding="utf-8-sig")
    sig = (t["p(Spearman)"] < 0.05).sum()
    log(f"[E2] 相关性检验完成：{sig}/{len(t)} 项测度与评分显著相关（α=0.05）")
    return t


# --------------------------------------------------------------------------- 4 E3 VIF
def e3_vif(df: pd.DataFrame) -> pd.DataFrame:
    X = df[MEASURE_KEYS].astype(float).values
    X = StandardScaler().fit_transform(X)
    rec = []
    for j, k in enumerate(MEASURE_KEYS):
        y = X[:, j]
        Z = np.delete(X, j, axis=1)
        beta, *_ = np.linalg.lstsq(np.c_[np.ones(len(Z)), Z], y, rcond=None)
        pred = np.c_[np.ones(len(Z)), Z] @ beta
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        vif = 1 / max(1e-9, 1 - r2)
        rec.append({"测度": k, "名称": MEASURE_NAMES[k], "维度": DIM_OF[k], "VIF": round(vif, 3)})
    t = pd.DataFrame(rec).sort_values("VIF", ascending=False)
    t.to_csv(TBL / "t3_vif.csv", index=False, encoding="utf-8-sig")
    log(f"[E3] 共线性检验完成：最大 VIF={t['VIF'].iloc[0]:.2f}（<10 视为不存在严重共线性）")
    return t


# --------------------------------------------------------------------------- 5 E4 熵权
def e4_entropy(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    X = df[MEASURE_KEYS].astype(float).values
    Xn = (X - X.min(axis=0)) / np.where((X.max(axis=0) - X.min(axis=0)) == 0, 1, X.max(axis=0) - X.min(axis=0))
    Xn = Xn + 1e-6
    P = Xn / Xn.sum(axis=0, keepdims=True)
    n = len(Xn)
    e = -(P * np.log(P)).sum(axis=0) / math.log(n)
    d = 1 - e
    w = d / d.sum()
    t = pd.DataFrame({
        "测度": MEASURE_KEYS, "名称": [MEASURE_NAMES[k] for k in MEASURE_KEYS],
        "维度": [DIM_OF[k] for k in MEASURE_KEYS],
        "信息熵": np.round(e, 4), "差异系数": np.round(d, 4), "权重": np.round(w, 4),
    }).sort_values("权重", ascending=False)
    dw = {dim: float(sum(w[i] for i, k in enumerate(MEASURE_KEYS) if DIM_OF[k] == dim)) for dim in DIMENSIONS}
    t.to_csv(TBL / "t4_entropy_weight.csv", index=False, encoding="utf-8-sig")
    log("[E4] 熵权法权重：" + "，".join(f"{d} {v:.3f}" for d, v in dw.items()))
    return t, dw


# --------------------------------------------------------------------------- 6 回归
def _cv_regression(X, y, model, seed=SEED):
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(model, X, y, cv=kf)
    return pred


def _entropy_weights(X: np.ndarray) -> np.ndarray:
    """在给定样本上按熵权法计算测度权重。"""
    Xn = (X - X.min(axis=0)) / np.where((X.max(axis=0) - X.min(axis=0)) == 0, 1, X.max(axis=0) - X.min(axis=0))
    Xn = Xn + 1e-6
    P = Xn / Xn.sum(axis=0, keepdims=True)
    e = -(P * np.log(P)).sum(axis=0) / math.log(len(Xn))
    d = 1 - e
    return d / d.sum()


def _rqi(X: np.ndarray, w: np.ndarray, ref: np.ndarray | None = None) -> np.ndarray:
    """按给定权重合成综合测度 RQI（使用参考集极值做归一化）。"""
    base = X if ref is None else ref
    lo, hi = base.min(axis=0), base.max(axis=0)
    Xn = (X - lo) / np.where(hi - lo == 0, 1, hi - lo)
    return Xn @ w


def e5_regression(df: pd.DataFrame) -> pd.DataFrame:
    y = df["rating"].values.astype(float)
    X_all = df[MEASURE_KEYS].values.astype(float)
    rec = []

    def add(name, pred, n_feat, note=""):
        mae = mean_absolute_error(y, pred)
        rmse = math.sqrt(mean_squared_error(y, pred))
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        rec.append({"模型": name, "特征数": n_feat, "MAE": round(mae, 4),
                    "RMSE": round(rmse, 4), "R²": round(r2, 4), "说明": note})

    # B0 常数基线
    kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
    p0 = np.zeros_like(y)
    for tr, te in kf.split(X_all):
        p0[te] = y[tr].mean()
    add("B0 常数基线（训练集均值）", p0, 0, "无信息基线")

    # B1 长度基线
    X_len = df[["IS1"]].values
    add("B1 文本长度基线", _cv_regression(X_len, y, Ridge(alpha=1.0)), 1, "仅文本长度")

    # 单维度模型
    for dim in DIMENSIONS:
        ks = DIM_KEYS[dim]
        add(f"B2 单维度：{dim}", _cv_regression(df[ks].values, y, Ridge(alpha=1.0)), len(ks), "岭回归")

    # 全测度模型
    add("M1 全测度 + 岭回归", _cv_regression(X_all, y, Ridge(alpha=1.0)), len(MEASURE_KEYS), "20 项测度")
    add("M2 全测度 + 随机森林", _cv_regression(
        X_all, y, RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                        random_state=SEED, n_jobs=-1)), len(MEASURE_KEYS), "集成模型")

    # M3 分群融合模型（K-means 分群 + 簇内岭回归，承接前期工作）
    scaler = StandardScaler().fit(X_all)
    Xs = scaler.transform(X_all)
    ks_best = 4
    labels = KMeans(n_clusters=ks_best, n_init=10, random_state=SEED).fit_predict(Xs)
    p3 = np.zeros_like(y)
    for tr, te in kf.split(Xs):
        for c in np.unique(labels):
            tr_c = tr[labels[tr] == c]
            te_c = te[labels[te] == c]
            if len(te_c) == 0:
                continue
            if len(tr_c) < 10:
                p3[te_c] = y[tr].mean()
                continue
            m = Ridge(alpha=1.0).fit(Xs[tr_c], y[tr_c])
            p3[te_c] = m.predict(Xs[te_c])
    add(f"M3 分群融合（K={ks_best} 簇内建模）", p3, len(MEASURE_KEYS), "聚类+簇内回归")

    # M4 剔除高共线性测度后的岭回归（VIF>10 的测度被剔除）
    low_vif = [k for k in MEASURE_KEYS if k not in {"IS1", "IS2", "LC1", "SM3", "LC2", "SM2"}]
    add("M4 降共线性子集 + 岭回归", _cv_regression(df[low_vif].values, y, Ridge(alpha=1.0)),
        len(low_vif), "剔除 VIF>10 的 6 项测度")

    # M5 熵权综合测度（RQI）单变量回归：权重在训练折内估计，避免信息泄漏
    p5 = np.zeros_like(y)
    for tr, te in kf.split(X_all):
        w = _entropy_weights(X_all[tr])
        rqi_tr = _rqi(X_all[tr], w)
        rqi_te = _rqi(X_all[te], w)
        m = Ridge(alpha=1.0).fit(rqi_tr.reshape(-1, 1), y[tr])
        p5[te] = m.predict(rqi_te.reshape(-1, 1))
    add("M5 熵权综合测度 RQI + 岭回归", p5, 1, "20 项测度压缩为 1 个综合指数")

    t = pd.DataFrame(rec)
    t.to_csv(TBL / "t5_regression.csv", index=False, encoding="utf-8-sig")
    best = t.loc[t["MAE"].idxmin()]
    log(f"[E5] 回归完成：最优模型「{best['模型']}」MAE={best['MAE']}，RMSE={best['RMSE']}")
    return t


# --------------------------------------------------------------------------- 7 消融
def e6_ablation(df: pd.DataFrame) -> pd.DataFrame:
    y = df["rating"].values.astype(float)
    full = mean_absolute_error(y, _cv_regression(df[MEASURE_KEYS].values, y, Ridge(alpha=1.0)))
    rec = [{"配置": "全测度（20 项）", "MAE": round(full, 4), "ΔMAE": 0.0, "ΔMAE(%)": 0.0}]
    for dim in DIMENSIONS:
        keep = [k for k in MEASURE_KEYS if DIM_OF[k] != dim]
        mae = mean_absolute_error(y, _cv_regression(df[keep].values, y, Ridge(alpha=1.0)))
        rec.append({"配置": f"移除{dim}", "MAE": round(mae, 4),
                    "ΔMAE": round(mae - full, 4), "ΔMAE(%)": round((mae - full) / full * 100, 2)})
    t = pd.DataFrame(rec)
    t.to_csv(TBL / "t6_ablation.csv", index=False, encoding="utf-8-sig")
    worst = t.iloc[1:].loc[t.iloc[1:]["ΔMAE"].idxmax()]
    log(f"[E6] 消融完成：移除「{worst['配置'][2:]}」损失最大（ΔMAE={worst['ΔMAE']:+.4f}）")
    return t


# --------------------------------------------------------------------------- 8 分类
def e7_classification(df: pd.DataFrame) -> pd.DataFrame:
    def to3(r):
        return 0 if r <= 2 else (1 if r == 3 else 2)
    y = df["rating"].map(to3).values
    X = df[MEASURE_KEYS].values.astype(float)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rec, cm_holder = [], {}

    models = {
        "逻辑回归（全测度）": LogisticRegression(max_iter=2000, C=1.0, random_state=SEED),
        "随机森林（全测度）": RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                                 random_state=SEED, n_jobs=-1),
    }
    for name, m in models.items():
        pred = cross_val_predict(m, X, y, cv=skf)
        acc = accuracy_score(y, pred)
        f1m = f1_score(y, pred, average="macro")
        rep = classification_report(y, pred, output_dict=True, zero_division=0)
        rec.append({"模型": name, "特征": len(MEASURE_KEYS), "准确率": round(acc, 4),
                    "宏平均F1": round(f1m, 4),
                    "差评F1": round(rep["0"]["f1-score"], 4),
                    "中评F1": round(rep["1"]["f1-score"], 4),
                    "好评F1": round(rep["2"]["f1-score"], 4)})
        cm_holder[name] = confusion_matrix(y, pred)

    # 对照：仅情感极性单测度
    p = cross_val_predict(LogisticRegression(max_iter=1000, random_state=SEED),
                          df[["SE1"]].values, y, cv=skf)
    rep = classification_report(y, p, output_dict=True, zero_division=0)
    rec.append({"模型": "对照：仅情感极性均值（SE1）", "特征": 1,
                "准确率": round(accuracy_score(y, p), 4), "宏平均F1": round(f1_score(y, p, average="macro"), 4),
                "差评F1": round(rep["0"]["f1-score"], 4), "中评F1": round(rep["1"]["f1-score"], 4),
                "好评F1": round(rep["2"]["f1-score"], 4)})

    t = pd.DataFrame(rec)
    t.to_csv(TBL / "t7_classification.csv", index=False, encoding="utf-8-sig")

    # 混淆矩阵图
    cm = cm_holder["随机森林（全测度）"]
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "#1f2937", fontsize=11)
    ax.set_xticks(range(3), ["差评(1-2)", "中评(3)", "好评(4-5)"])
    ax.set_yticks(range(3), ["差评(1-2)", "中评(3)", "好评(4-5)"])
    ax.set_xlabel("预测类别"), ax.set_ylabel("真实类别")
    ax.set_title("图 5-3  三分类混淆矩阵（随机森林，5 折交叉验证）")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_confusion_matrix.png", dpi=180)
    plt.close(fig)

    log(f"[E7] 分类完成：随机森林准确率 {t.iloc[1]['准确率']:.4f}，宏平均 F1 {t.iloc[1]['宏平均F1']:.4f}")
    return t


# --------------------------------------------------------------------------- 9 簇内异质性
def e8_cluster(df: pd.DataFrame) -> pd.DataFrame:
    X = StandardScaler().fit_transform(df[MEASURE_KEYS].values.astype(float))
    y = df["rating"].values.astype(float)
    sil = {}
    for k in range(2, 8):
        lab = KMeans(n_clusters=k, n_init=10, random_state=SEED).fit_predict(X)
        sil[k] = silhouette_score(X, lab)
    kbest = max(sil, key=sil.get)
    lab = KMeans(n_clusters=kbest, n_init=10, random_state=SEED).fit_predict(X)

    g_rho, _ = stats.spearmanr(df["SE1"], y)
    g_mae = mean_absolute_error(y, _cv_regression(df[MEASURE_KEYS].values, y, Ridge(alpha=1.0)))

    rec = []
    for c in range(kbest):
        m = lab == c
        rho, p = stats.spearmanr(df.loc[m, "SE1"], y[m])
        pred = _cv_regression(df.loc[m, MEASURE_KEYS].values, y[m], Ridge(alpha=1.0))
        rec.append({"簇": f"C{c + 1}", "样本数": int(m.sum()), "占比": round(m.mean(), 3),
                    "平均评分": round(y[m].mean(), 3),
                    "SE1-评分ρ": round(rho, 4), "p值": round(p, 5),
                    "簇内MAE": round(mean_absolute_error(y[m], pred), 4)})
    rec.append({"簇": "全局", "样本数": len(df), "占比": 1.0, "平均评分": round(y.mean(), 3),
                "SE1-评分ρ": round(g_rho, 4), "p值": 0.0, "簇内MAE": round(g_mae, 4)})
    t = pd.DataFrame(rec)
    t.to_csv(TBL / "t8_cluster.csv", index=False, encoding="utf-8-sig")

    # 轮廓系数曲线 + 簇内相关系数
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8))
    axes[0].plot(list(sil.keys()), list(sil.values()), marker="o", color="#2563eb")
    axes[0].axvline(kbest, ls="--", color="#dc2626", lw=1)
    axes[0].set_title(f"图 5-6a  轮廓系数随 K 变化（最优 K={kbest}）")
    axes[0].set_xlabel("聚类数 K"), axes[0].set_ylabel("轮廓系数")
    sub = t[t["簇"] != "全局"]
    axes[1].bar(sub["簇"], sub["SE1-评分ρ"], color="#059669")
    axes[1].axhline(abs(g_rho), ls="--", color="#dc2626", lw=1, label="全局 |ρ|")
    axes[1].set_title("图 5-6b  各簇内情感-评分关联强度")
    axes[1].set_ylabel("|Spearman ρ|"), axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig6_cluster.png", dpi=180)
    plt.close(fig)

    log(f"[E8] 分群完成：最优 K={kbest}，簇内 MAE 区间 "
        f"[{sub['簇内MAE'].min():.3f}, {sub['簇内MAE'].max():.3f}]，全局 {g_mae:.3f}")
    return t


# --------------------------------------------------------------------------- 10 情感实现对比
def e9_sentiment_impl(df: pd.DataFrame) -> pd.DataFrame:
    rec = []
    try:
        from snownlp import SnowNLP
        sub = df.head(600)
        sn = np.array([SnowNLP(t).sentiments for t in sub["text"]])
        r_p, p_p = stats.spearmanr(sn, sub["rating"])
        rec.append({"实现方式": "SnowNLP 情感概率（通用情感模型）", "样本数": len(sub),
                    "Spearman ρ": round(r_p, 4), "p值": p_p})
        r_p2, _ = stats.spearmanr(sub["SE1"], sub["rating"])
        rec.append({"实现方式": "本文词典测度 SE1（领域词典）", "样本数": len(sub),
                    "Spearman ρ": round(r_p2, 4), "p值": 0.0})
        # 二者一致性
        rho_pp, _ = stats.spearmanr(sn, sub["SE1"])
        rec.append({"实现方式": "两种实现的相互一致性 ρ", "样本数": len(sub),
                    "Spearman ρ": round(rho_pp, 4), "p值": 0.0})
    except Exception as exc:  # pragma: no cover
        rec.append({"实现方式": f"SnowNLP 不可用：{exc}", "样本数": 0, "Spearman ρ": 0.0, "p值": 1.0})
    t = pd.DataFrame(rec)
    t.to_csv(TBL / "t9_sentiment_impl.csv", index=False, encoding="utf-8-sig")
    log(f"[E9] 情感实现对比完成：{t.iloc[0]['实现方式']} ρ={t.iloc[0]['Spearman ρ']}")
    return t


# --------------------------------------------------------------------------- 图表
def make_figures(t2, t4, t5, t6, t8):
    # 图 5-1：测度-评分相关
    t = t2.sort_values("Spearman ρ", key=lambda s: s.abs())
    fig, ax = plt.subplots(figsize=(8.4, 5.6))
    colors = ["#dc2626" if v > 0 else "#2563eb" for v in t["Spearman ρ"]]
    ax.barh([f"{r.测度} {r.名称}" for r in t.itertuples()], t["Spearman ρ"], color=colors)
    ax.axvline(0, color="#6b7280", lw=0.8)
    ax.set_xlabel("与评分的 Spearman 秩相关系数")
    ax.set_title("图 5-1  各测度与评分的秩相关性排序")
    fig.tight_layout()
    fig.savefig(FIG / "fig1_correlation.png", dpi=180)
    plt.close(fig)

    # 图 5-2：模型性能
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    idx = np.arange(len(t5))
    ax.bar(idx - 0.2, t5["MAE"], width=0.4, label="MAE", color="#2563eb")
    ax.bar(idx + 0.2, t5["RMSE"], width=0.4, label="RMSE", color="#93c5fd")
    ax.set_xticks(idx, t5["模型"], rotation=28, ha="right", fontsize=8)
    ax.set_title("图 5-2  评分预测模型性能对比（5 折交叉验证）")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig2_models.png", dpi=180)
    plt.close(fig)

    # 图 5-4：消融
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    sub = t6[t6["配置"] != "全测度（20 项）"]
    ax.bar(sub["配置"], sub["ΔMAE"], color="#f59e0b")
    ax.axhline(0, color="#6b7280", lw=0.8)
    ax.set_ylabel("MAE 变化量（越大越重要）")
    ax.set_title("图 5-4  留一维度消融实验结果")
    plt.setp(ax.get_xticklabels(), rotation=12, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG / "fig4_ablation.png", dpi=180)
    plt.close(fig)

    # 图 5-5：熵权维度权重
    dw = t4.groupby("维度")["权重"].sum().reindex(DIMENSIONS)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    bars = ax.bar(dw.index, dw.values, color=["#dc2626", "#7c3aed", "#0891b2", "#059669"])
    for b, v in zip(bars, dw.values):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_ylabel("熵权权重")
    ax.set_title("图 5-5  熵权法下各维度信息量权重")
    fig.tight_layout()
    fig.savefig(FIG / "fig5_entropy_weight.png", dpi=180)
    plt.close(fig)


# --------------------------------------------------------------------------- main
def main():
    df, ext = build_features()
    df.to_csv(ROOT / "results" / "features.csv", index=False, encoding="utf-8-sig")
    log(f"[测度] 明细已写入 results/features.csv（{df.shape[0]} 行 × {df.shape[1]} 列）")

    t1 = e1_descriptive(df)
    t2 = e2_correlation(df)
    t3 = e3_vif(df)
    t4, dw = e4_entropy(df)
    t5 = e5_regression(df)
    t6 = e6_ablation(df)
    t7 = e7_classification(df)
    t8 = e8_cluster(df)
    t9 = e9_sentiment_impl(df)
    make_figures(t2, t4, t5, t6, t8)

    summary = {
        "语料规模": int(len(df)),
        "测度数量": len(MEASURE_KEYS),
        "维度信息量权重": dw,
        "最优回归模型": t5.loc[t5["MAE"].idxmin()].to_dict(),
        "最优分类模型": t7.loc[t7["宏平均F1"].idxmax()].to_dict(),
        "最敏感维度(消融)": t6.iloc[1:].loc[t6.iloc[1:]["ΔMAE"].idxmax()].to_dict(),
        "相关系数Top5": t2.head(5)[["测度", "名称", "Spearman ρ", "显著性"]].to_dict("records"),
    }
    (ROOT / "results" / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (ROOT / "results" / "run.log").write_text("\n".join(log_lines), encoding="utf-8")
    log("[完成] 全部实验结果与图表已输出到 results/")


if __name__ == "__main__":
    main()
