# -*- coding: utf-8 -*-
"""DBR-3K 中文图书评论语料构建脚本。

语料说明（对应论文 3.3 节）
---------------------------
DBR-3K（Douban-Book-Review-3K）为本研究构建的控制变量语料：
每条记录 = {review_id, book_title, category, rating(1~5), text, n_sent, tokens}

构建方式：以图书评论领域的真实语用单元（clause）为原子，按「评分 → 情感配置 →
语用槽位填充」的流程组合成完整评论。评分标签由生成过程真值给定，因此测度与评分
之间的关联强度是可控、可复现的，适合用于验证测度体系的有效性与建模性能。

局限：语言多样性弱于真实平台语料，绝对性能不代表真实场景；论文 6.2 节给出了
用真实数据复现的脚本接口（本脚本 + 测度库可直接接入任意 CSV 语料）。

运行：
    python corpus_builder.py            # 生成 data/reviews.csv（默认 3000 条）
    python corpus_builder.py --n 500    # 指定规模
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

# ---------------------------------------------------------------------------
# 语料原子：按「语用槽位 × 极性」组织
# ---------------------------------------------------------------------------
OPEN = {
    "neu": [
        "这本书是朋友推荐给我的，趁着周末读完。",
        "在图书馆偶然翻到，就借回来看了看。",
        "看榜单时注意到这本，顺手加进了书单。",
        "同事桌上放着这本，我借来读了两周。",
        "趁着活动入手，放了很久才开始读。",
        "这是我今年读的第三本同类型作品。",
        "看完相关的纪录片之后，想找本系统一点的书。",
        "之前在书店翻过几页，这次终于买回来读完。",
        "选这本书主要是因为它的目录结构看起来比较清楚。",
        "读这本书的契机是想补一补这个领域的基础知识。",
        "在旧书摊淘到的版本，纸张有些发黄。",
        "读之前对这本的期待值是中等偏上。",
    ],
    "pos": [
        "冲着豆瓣的高分买的，果然没有让我失望。",
        "早就听人夸过这本，抱着期待打开，结果比我预想的还好。",
        "被书评区安利了很多次，终于入手，读完只想说值回票价。",
        "封面和简介都很吸引我，翻开之后发现内容比简介更扎实。",
        "本来只是随手翻翻，没想到第一页就把我留住了。",
        "这本书在我的书架上躺了半年，这次一口气读完，后悔没早点看。",
    ],
    "neg": [
        "被封面和腰封上的宣传吸引，结果落差有点大。",
        "这本书在各大榜单上挂了很久，读完却有点名不副实的味道。",
        "因为评分很高才入手，读到一半就开始怀疑是不是看的同一本。",
        "买这本书的时候期待值挺高，读完只能说期望越大失望越大。",
        "看了不少推荐才下单，翻开之后很快就明白了问题所在。",
        "冲着作者的名气买的，读完之后心情比较复杂。",
    ],
}

CONTENT = {
    "pos": [
        "情节的设计很巧妙，几条线索收束得很干净。",
        "故事推进得很有章法，每一章都在往前推，没有多余的段落。",
        "作者把复杂的问题拆解得很清楚，读起来思路一直很顺。",
        "书里的论证层次分明，递进关系处理得相当漂亮。",
        "核心观点虽然不算新，但论述过程非常扎实，站得住脚。",
        "内容密度很高，几乎每一页都有可以记下来的东西。",
        "作者对细节的把握很到位，很多地方的处理让人眼前一亮。",
        "整体框架立得住，前后呼应的地方做得很用心。",
        "把抽象的道理落到具体的案例上，读起来一点都不觉得空。",
        "信息量很大但组织得有序，读完之后脑子里有一条清晰的主线。",
        "对争议问题没有回避，正反两方面的材料都给得很公平。",
        "故事里有好几处伏笔，最后都收回来了，这种完成度很少见。",
    ],
    "neg": [
        "情节推进得很慢，读到三分之一还在原地打转。",
        "很多章节像是为了凑字数，删掉一半内容也不会影响理解。",
        "论证过程比较跳跃，关键的地方往往一笔带过。",
        "核心观点反复重复，同一个意思翻来覆去讲了很多遍。",
        "案例选得比较随意，跟前面提出的框架对不上。",
        "结构松散，几部分之间缺少必要的过渡。",
        "内容水得比较厉害，真正有用的部分大概只有一成。",
        "逻辑链条中间断了好几处，作者似乎默认读者能自己补上。",
        "把简单的道理包装得很复杂，读完发现并没有说什么新东西。",
        "材料堆砌得很明显，缺少自己的分析和判断。",
        "同一段材料在书里出现了三次，编辑显然没有认真校过。",
        "概念前后不一致，读到后面已经分不清作者到底想说什么。",
    ],
}

CHARACTER = {
    "pos": [
        "人物写得很立体，缺点和优点都摆在明面上，很真实。",
        "主人公的成长线铺得很自然，转变的每一步都有铺垫。",
        "配角也没有被工具化，每个人都有自己的动机和逻辑。",
        "对话写得特别有生活气息，人物的性格全靠对话立住了。",
        "作者对人物心理的刻画很细腻，很多微妙的情绪都写到了。",
        "群像戏处理得相当好，几个主要角色都能在脑子里留下印象。",
        "人物的选择都符合各自的处境，没有为了推动剧情而强行降智。",
    ],
    "neg": [
        "人物扁平得像纸片，从头到尾没有变化。",
        "主角的行为动机站不住，很多决定完全是作者为了让剧情走下去硬塞的。",
        "配角的存在感很弱，基本只是报幕工具。",
        "对话写得比较生硬，不像真人会说出来的话。",
        "人物关系混乱，读了一半还没搞清楚谁和谁是什么关系。",
        "心理描写很表面，几乎都是贴着标签在写。",
        "角色的性格前后矛盾，前后像是两个人。",
    ],
}

STYLE = {
    "pos": [
        "文笔很干净，句子短但有力，读起来很舒服。",
        "语言非常克制，情绪都压在水面下，反而更有力量。",
        "行文流畅自然，长句和短句的节奏搭配得很好。",
        "作者的文字有辨识度，很多句子读完会想再读一遍。",
        "叙述视角切换得很顺，几乎感觉不到切换的痕迹。",
        "用词准确，很少看到凑数的形容词。",
        "把专业内容写得通俗易懂，外行也能读下去。",
        "语言平实但有味道，不炫技，反而显得诚恳。",
    ],
    "neg": [
        "文笔比较粗糙，句子之间缺少打磨的痕迹。",
        "语言晦涩，很多段落要读三遍才能大致明白意思。",
        "形容词堆砌得很厉害，读起来有种用力过猛的感觉。",
        "句式单一，读久了会觉得单调。",
        "文风前后不统一，前半本和后半本像是两个人写的。",
        "表达绕来绕去，一句话能说清的事情要用一段。",
        "文字有一股很浓的说教腔，读起来不太舒服。",
    ],
}

THEME = {
    "pos": [
        "提出的问题很有价值，读完会让人重新思考一些习以为常的判断。",
        "观点很有穿透力，能从表面的现象直接切到背后的机制。",
        "这本书提供的视角我在别处没有见过，光凭这一点就值得一读。",
        "主题虽然沉重，但作者的处理方式给了人希望。",
        "把个人经验放到更大的结构里去看，这一点做得很出色。",
        "思想上的启发很大，读完有几天一直在想书里提到的问题。",
    ],
    "neg": [
        "观点比较陈旧，很多结论在别的书里已经讲得更清楚。",
        "立场过于鲜明，为了论证自己的观点选择性地使用了材料。",
        "对复杂问题的处理过于简单化，给出了一个看起来很省事的答案。",
        "整本书的价值观输出有点强，读起来有些不适。",
        "所谓的新观点其实是把老话换了个说法。",
        "观点和材料之间有明显的缝隙，结论下得太快。",
    ],
}

TRANSLATION = {
    "pos": [
        "译文质量很高，读起来几乎感觉不到翻译腔。",
        "译者的处理很用心，注释加得恰到好处。",
        "专有名词的译法前后统一，读起来没有障碍。",
        "原文的语感保留得不错，句子读起来是有节奏的。",
    ],
    "neg": [
        "翻译痕迹很重，很多句子明显是照着原文的语序硬搬过来的。",
        "术语的译法前后不一致，读的时候要不停地在脑子里换算。",
        "错别字和漏译的地方不少，这个版本算不上合格。",
        "有几处明显的误译，会对理解原文造成误导。",
    ],
}

PRINTING = {
    "pos": [
        "装帧很用心，纸张手感很好，排版也舒服。",
        "开本大小合适，字距行距都调得不错，长时间读眼睛不累。",
        "内页的留白给得很大方，阅读体验很好。",
        "印刷清晰，图版的还原度很高。",
    ],
    "neg": [
        "纸张偏薄，背面的字会透过来。",
        "排版太密，行距很小，看久了眼睛发酸。",
        "字体偏小，长时间阅读比较吃力。",
        "装订质量一般，翻到中间几页有脱胶的迹象。",
    ],
}

PACE = {
    "pos": [
        "节奏把握得很好，张弛有度，几乎不觉得冷场。",
        "该快的地方快，该慢的地方肯停下来写细，分寸感很好。",
        "读起来很顺畅，两天的时间就翻完了。",
        "转折来得不突兀，每处变化都在意料之外情理之中。",
    ],
    "neg": [
        "中段明显拖沓，有几章几乎可以整段跳过。",
        "节奏忽快忽慢，后半部分匆匆收尾。",
        "铺垫太久，高潮部分却没有相应的分量。",
        "读完前半本的热情在中间被消磨得差不多了。",
    ],
}

PRICE = {
    "pos": [
        "这个价格能买到这样的内容，性价比很高。",
        "属于买回来会反复翻的那类书，不算贵。",
        "趁折扣入手，感觉很划算。",
    ],
    "neg": [
        "定价偏高，内容撑不起这个厚度。",
        "同样的内容在别家便宜不少，这个版本不太值得。",
        "花这个价钱买回来，心里有点不平衡。",
    ],
}

ENDING = {
    "pos": [
        "总的来说，这是我近期读到的最满意的一本，推荐给同好。",
        "整体质量在线，会找作者的其他作品来读。",
        "如果题材对你胃口，这本值得放进书单。",
        "五星给得毫不犹豫，希望有更多人读到。",
        "读完之后在书里做了不少标注，之后还会再翻。",
    ],
    "neg": [
        "综合来看不太推荐，时间有限的读者可以跳过。",
        "两星是给内容的，剩下的一星留给装帧。",
        "如果只是想了解这个领域，别的书可能更合适。",
        "读完有些遗憾，这本更像是半成品。",
        "不会再读第二遍，也不会推荐给朋友。",
    ],
    "neu": [
        "总体中规中矩，有用但谈不上惊喜。",
        "优点和缺点都很明显，看个人需求吧。",
        "作为入门读物还可以，深入的话需要另找资料。",
        "三颗星，属于读完就放下、不会特别记住的那类。",
    ],
}

CONNECTORS = ["不过，", "当然，", "另外，", "话说回来，", "值得一提的是，",
              "相比之下，", "除此之外，", "客观地说，", "坦白讲，", "整体而言，"]

# 评分 → 语用槽位与情感配置
# 每个评分档给出：槽位池、各槽位取正/负情感的比例、句数范围
PROFILE = {
    5: {"slots": ["content", "character", "style", "theme", "pace", "price", "translation", "printing"],
        "p_pos": 0.94, "n_range": (3, 5)},
    4: {"slots": ["content", "character", "style", "theme", "pace", "translation"],
        "p_pos": 0.80, "n_range": (3, 5)},
    3: {"slots": ["content", "character", "style", "theme", "pace", "printing"],
        "p_pos": 0.50, "n_range": (2, 4)},
    2: {"slots": ["content", "character", "style", "pace", "translation", "printing", "price"],
        "p_pos": 0.20, "n_range": (2, 4)},
    1: {"slots": ["content", "character", "style", "theme", "pace", "printing", "price", "translation"],
        "p_pos": 0.08, "n_range": (2, 4)},
}

SLOT_MAP = {
    "content": CONTENT, "character": CHARACTER, "style": STYLE, "theme": THEME,
    "pace": PACE, "price": PRICE, "translation": TRANSLATION, "printing": PRINTING,
}

# 评分分布（贴近真实长尾分布：好评偏多，中间偏少）
RATING_WEIGHTS = {5: 0.36, 4: 0.28, 3: 0.16, 2: 0.11, 1: 0.09}


def build_one(rng: random.Random, rating: int, book: dict) -> str:
    cfg = PROFILE[rating]
    parts: list[str] = []

    # 1) 开头语（与评分同向的预期落空/被满足表达）
    if rating >= 4:
        parts.append(rng.choice(OPEN["pos"] if rng.random() < 0.72 else OPEN["neu"]))
    elif rating <= 2:
        parts.append(rng.choice(OPEN["neg"] if rng.random() < 0.72 else OPEN["neu"]))
    else:
        parts.append(rng.choice(OPEN["neu"]))

    # 2) 主体：按槽位抽样
    n = rng.randint(*cfg["n_range"])
    slots = rng.sample(cfg["slots"], k=min(n, len(cfg["slots"])))
    for slot in slots:
        pool = SLOT_MAP[slot]
        polarity = "pos" if rng.random() < cfg["p_pos"] else "neg"
        if slot in ("price", "translation", "printing") and polarity == "neg" and rng.random() < 0.35:
            continue  # 少量槽位缺省，避免句法过于整齐
        sent = rng.choice(pool[polarity])
        if len(parts) > 1 and rng.random() < 0.34:
            sent = rng.choice(CONNECTORS) + sent
        parts.append(sent)

    # 3) 结尾
    if rating >= 4:
        tail_pool = ENDING["pos"]
    elif rating <= 2:
        tail_pool = ENDING["neg"]
    else:
        tail_pool = ENDING["neu"] if rng.random() < 0.6 else (ENDING["pos"] if rng.random() < 0.5 else ENDING["neg"])
    parts.append(rng.choice(tail_pool))

    text = "".join(parts)
    # 4) 表层噪声：少量语气词/口语插入，增加语言多样性
    if rng.random() < 0.22:
        text = text.replace("。", rng.choice(["。", "。", "……", "！"]), 1)
    if rng.random() < 0.15:
        text = text.replace("总的来说", rng.choice(["总而言之", "总的说来", "综合起来"]), 1)
    if rng.random() < 0.10:
        text = text.replace("这本书", rng.choice(["这本", "本作", "该书"]), 1)
    return text


def build(n: int = 3000, seed: int = 20260517) -> list[dict]:
    rng = random.Random(seed)
    books = [
        ("人间失格", "外国文学"), ("百年孤独", "外国文学"), ("活着", "中国文学"),
        ("围城", "中国文学"), ("明朝那些事儿", "历史"), ("枪炮、病菌与钢铁", "社科"),
        ("人类简史", "社科"), ("时间简史", "科普"), ("自私的基因", "科普"),
        ("万历十五年", "历史"), ("解忧杂货店", "外国文学"), ("了不起的盖茨比", "外国文学"),
        ("苏东坡传", "传记"), ("叫魂", "历史"), ("焦虑的人", "外国文学"),
        ("认知觉醒", "心理"), ("学会提问", "社科"), ("如何阅读一本书", "方法"),
        ("被讨厌的勇气", "心理"), ("局外人", "外国文学"), ("乡土中国", "社科"),
        ("鱼王", "外国文学"), ("我是猫", "外国文学"), ("夜航西飞", "传记"),
    ]
    ratings = list(RATING_WEIGHTS.keys())
    weights = list(RATING_WEIGHTS.values())

    rows = []
    for i in range(n):
        rating = rng.choices(ratings, weights=weights, k=1)[0]
        title, cat = rng.choice(books)
        text = build_one(rng, rating, {"title": title, "category": cat})
        rows.append({
            "review_id": f"DBR{i + 1:05d}",
            "book_title": title,
            "category": cat,
            "rating": rating,
            "text": text,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=20260517)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "reviews.csv"))
    args = ap.parse_args()

    rows = build(args.n, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["review_id", "book_title", "category", "rating", "text"])
        w.writeheader()
        w.writerows(rows)

    from collections import Counter
    dist = Counter(r["rating"] for r in rows)
    lens = [len(r["text"]) for r in rows]
    print(f"已生成 {len(rows)} 条评论 -> {out}")
    print("评分分布：" + "，".join(f"{k}星={dist[k]}({dist[k]/len(rows):.1%})" for k in sorted(dist)))
    print(f"评论长度：均值 {sum(lens)/len(lens):.1f} 字，最短 {min(lens)}，最长 {max(lens)}")
    print(f"示例：{rows[0]['rating']} 星｜{rows[0]['text'][:80]}...")


if __name__ == "__main__":
    main()
