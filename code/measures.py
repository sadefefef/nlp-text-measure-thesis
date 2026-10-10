# -*- coding: utf-8 -*-
"""多维 NLP 文本测度体系（论文第 4 章的代码实现）。

共 4 个维度、20 项测度：

D1 情感维度 SE   : SE1 极性均值 / SE2 情感强度 / SE3 情感波动 / SE4 正负倾向度 / SE5 情感一致性
D2 语义维度 SM   : SM1 主题集中度 / SM2 信息含量 / SM3 领域相关度 / SM4 语义离散度
D3 语言复杂度 LC : LC1 词汇丰富度 / LC2 平均句长 / LC3 词汇难度 / LC4 长词率 / LC5 修饰密度
D4 信息结构 IS   : IS1 文本长度 / IS2 句数 / IS3 标点多样性 / IS4 标点密度 / IS5 信息密度 / IS6 具体性

所有测度均给出显式定义并可在论文中复现；除 SE、LC5 依赖情感词典外，其余测度
均为纯统计型测度，不依赖任何预训练模型。
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Tuple

import jieba
import jieba.posseg as pseg
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from lexicon import (SENTIMENT_LEXICON, DEGREE_ADVERBS, NEGATION_WORDS, STOPWORDS,
                     PUNCT_GROUPS, DOMAIN_TERMS, POSITIVE_WORDS, NEGATIVE_WORDS)

LEXICON_SIZE = {"positive": len(POSITIVE_WORDS), "negative": len(NEGATIVE_WORDS)}

jieba.setLogLevel(60)

SENT_END = "。！？!?…；;\n"
PUNCT_CHARS = set("".join(sum(PUNCT_GROUPS.values(), [])))
ALL_PUNCT = PUNCT_CHARS | set("，。！？；：、…—")

DIM_OF = {
    "SE1": "情感维度", "SE2": "情感维度", "SE3": "情感维度", "SE4": "情感维度", "SE5": "情感维度",
    "SM1": "语义维度", "SM2": "语义维度", "SM3": "语义维度", "SM4": "语义维度",
    "LC1": "语言复杂度", "LC2": "语言复杂度", "LC3": "语言复杂度", "LC4": "语言复杂度", "LC5": "语言复杂度",
    "IS1": "信息结构", "IS2": "信息结构", "IS3": "信息结构", "IS4": "信息结构",
    "IS5": "信息结构", "IS6": "信息结构",
}
DIMENSIONS = ["情感维度", "语义维度", "语言复杂度", "信息结构"]
MEASURE_NAMES = {
    "SE1": "情感极性均值", "SE2": "情感强度", "SE3": "情感波动", "SE4": "正负倾向度", "SE5": "情感一致性",
    "SM1": "主题集中度", "SM2": "信息含量", "SM3": "领域相关度", "SM4": "语义离散度",
    "LC1": "词汇丰富度", "LC2": "平均句长", "LC3": "词汇难度", "LC4": "长词率", "LC5": "修饰密度",
    "IS1": "文本长度", "IS2": "句数", "IS3": "标点多样性", "IS4": "标点密度",
    "IS5": "信息密度", "IS6": "具体性",
}
MEASURE_KEYS = list(MEASURE_NAMES.keys())

# ---------------- 中文常用字频（用于词汇难度测度，取自通用字频序，取前 3500 常用字） ----
_COMMON_HANZI = (
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
    "见明问力理尔点文几定本公特做外孩相西果走将月十实向声车全信重三机工物气每并别真打太新比才便夫再书部水像眼等体却加电主界门利海受听表德少克代员许稜先口由死安写性马光白或住难望教命花结乐色更拉东神记处让母父应直字场平报友关放至张认接告入笑内英军候民岁往何度山觉路带万男边风解叫任金快原吃妈变通师立象数四失满战远格士音轻目条呢病始达深完今提求清王化空业思切怎非找片罗钱紶吗语元喜曾离飞科言干流欢约各即指合反题必该论交终林请医晚制球决窢传画保读运及则房早院量苦火布品近坐产答星精视五连司巴奇管类未朋且婚台夜青北队久乎越观落尽形影红爸百令周吧识步希亚术留市半热送兴造谈容极随演收首根讲整式取照办强石古华諣拿计您装似足双妻尼转诉米称丽客南领节衣站黑刻统断福城故历惊脸选包紧争另建维绝树系伤示愿持千史谁准联妇纪基买志静阿诗独复痛消社算义竟确酒需单治卡幸兰念举仅钟怕共毛句息功官待究跟穿室易游程号居考突皮哪费倒价图具刚脑永歌响商礼细专黄块脚味灵改据般破引食仍存众注笔甚某沉血备习校默务土微娘须试怀料调广蜖苏显赛查密议底列富梦错座参八除跑亮假印设线温虽掉京初养香停际致阳纸李纳验助激够严证帝饭忘趣支春集丈木研班普导顿睡展跳获艺六波察群皇段急庭创区奥器谢弟店否害草排背止组州朝封睛板角况曲馆育忙质河续哥呼若推境遇雨标姐充围案伦护冷警贝著雪索剧啊船险烟依斗值帮汉慢佛肯闻唱沙局伯族低玩资屋击速顾泪洲团圣旁堂兵七露园牛哭旅街劳型烈姑陈莫鱼异抱宝权鲁简态级票怪寻杀律胜份汽右洋范床舞秘午登楼贵吸责例追较职属渐左录丝牙党继托赶章智冲叶胡吉卖坚喝肉遗救修松临藏担戏善卫药悲敢靠伊村戴词森耳差短祖云规窗散迷油旧适乡架恩投弹铁博雷府压超负勒杂醒洗采毫嘴毕九冰既状乱景席珍童顶派素脱农疑练野按犯拍征坏骨余承置臂彩灯巨琴免环姆暗换技翻束增忍餐洛塞缺忆判欧层付阵玛批岛项狗休懂武革良恶恋委拥娜妙探呀营退摇弄桌熟诺宣银势奖宫忽套康供优课鸟喊降夏困刘罪亡鞋健模败伴守挥鲜财孤枪禁恐伙杰迹妹藸遍盖副坦牌江顺秋萨菜划授归浪听凡预奶雄升碃编典袋莱含盛济蒙棋端腿招释介烧误"
)


class MeasureExtractor:
    """测度抽取器。需先在语料上 fit（估计 IDF、类别质心、句子 SVD 空间）。"""

    def __init__(self, n_svd: int = 24, random_state: int = 20260517):
        self.n_svd = n_svd
        self.random_state = random_state
        self.word_tfidf = TfidfVectorizer(
            tokenizer=lambda s: tokenize_words(s), lowercase=False,
            token_pattern=None, min_df=2, sublinear_tf=True,
        )
        self.char_freq = self._build_char_freq()
        self.idf_map: Dict[str, float] = {}
        self.category_centroid: Dict[str, np.ndarray] = {}
        self.svd: TruncatedSVD | None = None
        self.global_centroid: np.ndarray | None = None

    # ------------------------------------------------------------------ fit
    def fit(self, texts: List[str], categories: List[str]) -> "MeasureExtractor":
        X = self.word_tfidf.fit_transform(texts)
        vocab = self.word_tfidf.get_feature_names_out()
        idf = self.word_tfidf.idf_
        self.idf_map = {w: float(v) for w, v in zip(vocab, idf)}
        self._default_idf = float(np.max(idf)) if len(idf) else 1.0

        self.global_centroid = np.asarray(X.mean(axis=0)).ravel()
        for cat in sorted(set(categories)):
            idx = [i for i, c in enumerate(categories) if c == cat]
            self.category_centroid[cat] = np.asarray(X[idx].mean(axis=0)).ravel()

        # 句子级 SVD 空间：用于 SM4 语义离散度
        sents = [s for t in texts[:1500] for s in split_sentences(t) if len(s) >= 4]
        self.sent_vec = TfidfVectorizer(
            tokenizer=lambda s: tokenize_words(s), lowercase=False,
            token_pattern=None, min_df=1, sublinear_tf=True,
        ).fit(sents)
        n_comp = min(self.n_svd, max(2, self.sent_vec.transform(sents).shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_comp, random_state=self.random_state)
        self.svd.fit(self.sent_vec.transform(sents))
        return self

    # -------------------------------------------------------------- helper
    @staticmethod
    def _build_char_freq() -> Dict[str, float]:
        # 以通用常用字表的位置构造字频近似：越靠前越高频
        out = {}
        n = len(_COMMON_HANZI)
        for i, ch in enumerate(_COMMON_HANZI):
            out[ch] = (n - i) / n
        return out

    def _sent_polarity(self, sent: str) -> Tuple[float, float]:
        """返回 (极性 ∈[-1,1], 强度 ∈[0,1.8])，含否定与程度副词处理。"""
        words = tokenize_words(sent)
        if not words:
            return 0.0, 0.0
        score, hits = 0.0, 0
        i = 0
        while i < len(words):
            w = words[i]
            # 向前看 2 个 token 找修饰语
            mod, neg = 1.0, False
            for j in range(max(0, i - 2), i):
                if words[j] in DEGREE_ADVERBS:
                    mod *= DEGREE_ADVERBS[words[j]]
                if words[j] in NEGATION_WORDS:
                    neg = not neg
            if w in SENTIMENT_LEXICON:
                pol, strength = SENTIMENT_LEXICON[w]
                s = pol * strength * mod
                if neg:
                    s = -s * 0.85
                score += s
                hits += 1
            i += 1
        if hits == 0:
            return 0.0, 0.0
        raw = score / hits                      # 尚未归一
        polarity = math.tanh(raw / 0.9)          # 压缩到 [-1,1]
        intensity = min(abs(score) / max(1.0, hits * 1.4), 1.5)
        return polarity, intensity

    # --------------------------------------------------------------- 主入口
    def extract(self, text: str, category: str) -> Dict[str, float]:
        sents = split_sentences(text)
        words = tokenize_words(text)
        content_words = [w for w in words if w not in STOPWORDS and w not in ALL_PUNCT

                         and not w.isdigit()]
        n_word = max(1, len(words))
        n_char = max(1, len(text))

        # ---------------- D1 情感维度 ----------------
        pols, ints = [], []
        for s in sents:
            p, it = self._sent_polarity(s)
            pols.append(p)
            ints.append(it)
        pos_cnt = sum(1 for p in pols if p > 0.05)
        neg_cnt = sum(1 for p in pols if p < -0.05)
        SE1 = float(np.mean(pols)) if pols else 0.0
        SE2 = float(np.mean([i for i in ints])) if ints else 0.0
        SE3 = float(np.std(pols)) if len(pols) > 1 else 0.0
        SE4 = float((pos_cnt - neg_cnt) / max(1, pos_cnt + neg_cnt))
        dom = 1 if SE1 >= 0 else -1
        SE5 = float(sum(1 for p in pols if (1 if p >= 0 else -1) == dom) / max(1, len(pols)))

        # ---------------- D2 语义维度 ----------------
        # SM1 主题集中度：评价侧面（评论维度词）分布的最大份额
        #     tot=0 视为无主题 -> 0；tot=1 视为完全集中于单一侧面 -> 1
        aspect = Counter(w for w in words if w in DOMAIN_TERMS)
        tot_aspect = sum(aspect.values())
        if tot_aspect >= 2:
            SM1 = float(max(aspect.values()) / tot_aspect)
        elif tot_aspect == 1:
            SM1 = 1.0
        else:
            SM1 = 0.0
        SM2 = float(np.mean([self.idf_map.get(w, self._default_idf) for w in content_words])) \
            if content_words else 0.0
        vec = self.word_tfidf.transform([text])
        dense = np.asarray(vec.todense()).ravel()
        centroid = self.category_centroid.get(category, self.global_centroid)
        SM3 = float(_cos(dense, centroid))
        SM4 = self._semantic_dispersion(sents)

        # ---------------- D3 语言复杂度 ----------------
        types = len(set(content_words))
        LC1 = float(types / math.sqrt(max(1, len(content_words))))
        sent_lens = [len(s) for s in sents if s.strip()]
        LC2 = float(np.mean(sent_lens)) if sent_lens else float(n_char)
        if content_words:
            diff = []
            for w in content_words:
                if len(w) == 1 and w in self.char_freq:
                    diff.append(-math.log10(max(self.char_freq[w], 1e-3)))
                else:
                    f = jieba.dt.FREQ.get(w, 10)
                    diff.append(-math.log10(max(f, 1.0) / 1e6))
            LC3 = float(np.mean(diff))
        else:
            LC3 = 0.0
        LC4 = float(sum(1 for w in content_words if len(w) >= 3) / max(1, len(content_words)))
        mod_cnt = 0
        for w, flag in pseg.cut(text):
            if flag.startswith(("a", "d")) or w in DEGREE_ADVERBS:
                mod_cnt += 1
        LC5 = float(mod_cnt / n_word)

        # ---------------- D4 信息结构 ----------------
        IS1 = float(math.log1p(n_char))
        IS2 = float(len([s for s in sents if s.strip()]))
        pc = Counter(ch for ch in text if ch in PUNCT_CHARS)
        if pc:
            q = np.array(list(pc.values()), dtype=float)
            q = q / q.sum()
            IS3 = float(-(q * np.log(q)).sum())
        else:
            IS3 = 0.0
        IS4 = float(sum(pc.values()) / n_char)
        IS5 = float(len(content_words) / n_word)
        num_tokens = sum(1 for w in words if any(c.isdigit() for c in w))
        quan = {"很", "多", "少", "几", "些", "个", "本", "页", "章", "年", "月", "天", "倍", "半", "遍"}
        quan_cnt = sum(1 for w in words if w in quan)
        IS6 = float((num_tokens + quan_cnt) / n_word)

        return {
            "SE1": SE1, "SE2": SE2, "SE3": SE3, "SE4": SE4, "SE5": SE5,
            "SM1": SM1, "SM2": SM2, "SM3": SM3, "SM4": SM4,
            "LC1": LC1, "LC2": LC2, "LC3": LC3, "LC4": LC4, "LC5": LC5,
            "IS1": IS1, "IS2": IS2, "IS3": IS3, "IS4": IS4, "IS5": IS5, "IS6": IS6,
        }

    def _semantic_dispersion(self, sents: List[str]) -> float:
        ss = [s for s in sents if len(s) >= 4]
        if len(ss) < 2:
            return 0.0
        X = self.svd.transform(self.sent_vec.transform(ss))
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1e-9
        Z = X / norms
        S = Z @ Z.T
        iu = np.triu_indices(len(ss), k=1)
        return float(1 - S[iu].mean())


# ---------------------------------------------------------------- 工具函数
def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(a @ b / (na * nb))


def tokenize_words(text: str) -> List[str]:
    return [w for w in jieba.lcut(text) if w.strip()]


def split_sentences(text: str) -> List[str]:
    out, buf = [], []
    for ch in text:
        buf.append(ch)
        if ch in SENT_END:
            s = "".join(buf).strip()
            if s:
                out.append(s)
            buf = []
    if buf:
        s = "".join(buf).strip()
        if s:
            out.append(s)
    return out
