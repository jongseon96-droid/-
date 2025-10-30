# -*- coding: utf-8 -*-
"""
[수정] Streamlit App v2.5 — TAB1 게이지바 가로 배치, TAB4 단어은행 Chi2로 복원
- TAB1: [수정] Plotly 게이지 차트(S/I)를 st.columns로 가로 배치, [신규] 목표 점수 텍스트로 강조
- TAB2: [수정] 토픽 단어 은행 생성 함수를 LogReg에서 다시 Chi2로 복원 (샘플 부족 문제 해결)
- TAB4: [수정] 단어 은행 설명을 Chi2 기준으로 복원
"""

import os, io, json, time, re as regx
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px # 시각화 추가
import plotly.graph_objects as go # [신규] 게이지 차트용
import datetime 
from typing import List, Dict, Tuple

# ==== OpenAI SDK ====
from openai import OpenAI
from openai import APIError, RateLimitError

# ==== ML / NLP ====
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle as sk_shuffle
from sklearn.feature_selection import chi2
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

# ==== 회귀 ====
import statsmodels.api as sm

# [신규] 한국어 불용어 리스트 정의
STOPWORDS_KO = [
    # 조사/어미 (매우 빈번)
    "입니다", "합니다", "같습니다", "있습니다", "있는", "것입니다", "했다", "등", "이", "그", "저",
    "수", "것", "및", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "에서", "으로", "하는", "입니다", "입니다", "입니다", "을", "를", "은", "는", "이", "가", "의", "에", "와", "과", "도",
    
    # 일반 명사 (신호 방해)
    "블로그", "포스팅", "오늘", "이번", "다양한", "관련", "내용", "정보", "정말", "바로", "지금", "생각",
    "경우", "대해", "대부분", "때문", "관련", "가지", "통해", "위해", "대한", "통한", "따라",
    
    # 스크린샷에서 보인 문제 단어들
    "2024", "2023", "ai", "2024년", "2023년", "자살", "극우", "ㅋㅋ", "ㅎㅎ"
]


# ================== CONFIG ==================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
USE_LLM = len(OPENAI_API_KEY) > 0
client = OpenAI(api_key=OPENAI_API_KEY) if USE_LLM else None
MODEL_CHAT = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini-2024-07-18")

LLM_OK = False
if USE_LLM and client:
    try:
        client.models.list()
        LLM_OK = True
    except Exception:
        LLM_OK = False

# ========= Robust CSV Loader =========
def read_csv_robust(src, **kwargs) -> pd.DataFrame:
    """UploadedFile/bytes/path/file-like 모두 지원. 인코딩과 구분자 자동 재시도."""
    encodings = ["utf-8", "utf-8-sig", "cp949", "euc-kr", "latin1"]
    seps = [None, ",", "\t", ";"]

    # bytes로 안전 복사
    if hasattr(src, "getvalue"):        # Streamlit UploadedFile
        raw = src.getvalue()
    elif isinstance(src, (bytes, bytearray)):
        raw = bytes(src)
    elif isinstance(src, str):          # 경로
        with open(src, "rb") as f:
            raw = f.read()
    else:                               # file-like
        raw = src.read()
        try:
            src.seek(0)
        except Exception:
            pass

    last_err = None
    for enc in encodings:
        for sep in seps:
            try:
                return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=sep, engine="python", **kwargs)
            except Exception as e:
                last_err = e
                continue
    raise RuntimeError(f"CSV 디코딩 실패: 마지막 오류={last_err}")

# ========= JSON 파서 =========
def _parse_json_safely(txt: str):
    """코드펜스/앞뒤 쓰레기/한글 BOM 제거, 첫 JSON 객체/배열만 파싱"""
    if not isinstance(txt, str):
        raise ValueError("LLM 응답이 비어있음")
    t = txt.strip().lstrip("\ufeff")
    if t.startswith("```"):
        parts = t.split("```")
        if len(parts) >= 3:
            cand = parts[1] if parts[1].strip().startswith(("{","[")) else parts[2]
            t = cand
        else:
            t = t.replace("```","").strip()
    si, sj = t.find("["), t.rfind("]")
    oi, oj = t.find("{"), t.rfind("}")
    if 0 <= si < sj:
        return json.loads(t[si:sj+1])
    if 0 <= oi < oj:
        return json.loads(t[oi:oj+1])
    raise ValueError("LLM 응답에서 JSON을 찾지 못함")

# ====== Candidate templates (폴백) ======
NUM_RE  = regx.compile(r"\b(\d+|top\s*\d+|[0-9]+분)\b", regx.I)
TIME_BANK = ["오늘", "이번 주", "주말", "지금", "방금", "이번 달", "10월", "11월", "12월"]
HOWTO_BANK = ["방-step", "베스트 프랙티스"]
ACTION_BANK = ["정리", "비교", "분석", "설명", "추천", "점검", "실법", "가이드", "체크리스트", "튜토리얼", "Step-by험"]
CTA_BANK = ["질문", "댓글", "구독", "공유", "알림", "참여"]
LIST_BANK = ["Top 5", "Top 7", "3가지", "5분 요약", "한눈에"]
BRAND_HINT = ["한양대", "오픈AI", "카카오", "구글", "MS", "네이버"]
DEFAULT_CANDIDATES = TIME_BANK + HOWTO_BANK + ACTION_BANK + CTA_BANK + LIST_BANK + BRAND_HINT

# ========= Utility =========
def categorize_term(t: str) -> str:
    t_low = t.lower()
    if NUM_RE.search(t_low) or any(x in t for x in LIST_BANK): return "숫자/리스트"
    if any(k in t for k in TIME_BANK): return "시간표현"
    if any(k in t for k in HOWTO_BANK): return "How-to/가이드"
    if any(k in t for k in CTA_BANK): return "질문/CTA"
    if any(k in t for k in ACTION_BANK): return "행동동사/행위"
    if regx.match(r"[A-Z][a-zA-Z0-9]+", t) or "대" in t or "대학" in t or any(b in t for b in BRAND_HINT):
        return "고유명사/브랜드"
    return "기타"

# ==== article_id 스키마 보정(최소 패치) ====
_CTRL = regx.compile(r"[\x00-\x1f\x7f]")

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명 BOM/공백 제거 + 소문자 + 스페이스→언더스코어"""
    df = df.copy()
    df.columns = (
        pd.Index(df.columns)
        .map(lambda c: str(c).lstrip("\ufeff").strip().lower().replace(" ", "_"))
    )
    return df

def coerce_article_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    - 컬럼명 정규화
    - article_id 별칭(id, doc_id, post_id 등)을 article_id로 통일
    - 값의 BOM/제어문자/공백 제거 + 문자열화
    """
    df = _normalize_columns(df.copy())
    aliases = ["article_id", "id", "doc_id", "post_id", "review_id", "news_id", "content_id"]
    found = None
    for a in aliases:
        if a in df.columns:
            found = a
            break
    if found is None:
        raise KeyError(f"CSV에 article_id 계열 컬럼이 없습니다. (컬럼={list(df.columns)[:12]})")
    if found != "article_id":
        df = df.rename(columns={found: "article_id"})

    df["article_id"] = (
        df["article_id"]
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .apply(lambda x: _CTRL.sub("", x))
        .str.strip()
    )
    return df

# ====== [수정] 토픽 단어 은행 (Chi2 기반으로 복원) ======
def build_topic_term_bank(df: pd.DataFrame,
                          vect: CountVectorizer,
                          lda_model: LatentDirichletAllocation,
                          topn: int = 80,
                          min_samples: int = 10) -> dict: # [수정] 최소 샘플 수 10
    """
    [수정] LogReg 대신 Chi2 (카이제곱) 통계로 복원 (샘플 부족 문제 해결)
    - 'good': 해당 토픽에서 'Good' 라벨과 통계적으로 가장 연관성이 높은 단어
    - 'bad': 해당 토픽에서 'Bad' 라벨과 통계적으로 가장 연관성이 높은 단어
    """
    bank = {}
    
    # [수정] 'topic' 컬럼이 없는 경우를 대비
    if 'topic' not in df.columns:
        return bank

    vocab = np.array(vect.get_feature_names_out())
    X = vect.transform((df["title"].fillna("") + " " + df["content"].fillna("")).tolist())
    topics = df["topic"].astype(int).values
    labels = (df["quality_label"] == "good").astype(int).values

    for t in sorted(np.unique(topics)):
        idx_t = np.where(topics == t)[0]
        
        # [수정] 샘플 수 체크 및 '실패 기록' 남기기
        if len(idx_t) < min_samples:
            bank[int(t)] = {
                "status": "error", 
                "message": f"샘플 부족 (N={len(idx_t)}, 최소 {min_samples} 필요)"
            }
            continue
            
        Xt = X[idx_t]
        yt = labels[idx_t]

        try:
            # 'Good' 라벨과 'Bad' 라벨이 모두 존재하는지 확인
            if len(np.unique(yt)) < 2:
                 bank[int(t)] = {"status": "error", "message": f"단일 라벨만 존재 (N={len(idx_t)})"}
                 continue

            chi_good, _ = chi2(Xt, yt)
            order_g = np.argsort(chi_good)[::-1]
            good_terms = [(vocab[i], float(chi_good[i])) for i in order_g[:topn]]

            chi_bad, _ = chi2(Xt, 1 - yt) # 'Bad' 라벨(1-yt)과 연관성
            order_b = np.argsort(chi_bad)[::-1]
            bad_terms = [(vocab[i], float(chi_bad[i])) for i in order_b[:topn]]

            counts = np.asarray(Xt.sum(axis=0)).ravel()
            order_all = np.argsort(counts)[::-1][:topn]
            all_terms = [(vocab[i], float(counts[i])) for i in order_all]

            bank[int(t)] = {
                "good": good_terms, 
                "bad": bad_terms, 
                "all": all_terms,
                "status": "ok", 
                "message": f"성공 (N={len(idx_t)})"
            }
        except Exception as e:
             bank[int(t)] = {"status": "error", "message": f"Chi2 계산 오류: {e}"}
    return bank


# ====== Draft → Topic 추론 ======
def infer_topic_for_text(txt: str,
                         vect: CountVectorizer,
                         lda_model: LatentDirichletAllocation) -> Tuple[int, np.ndarray]:
    Xd = vect.transform([txt if isinstance(txt, str) else ""])
    dist = lda_model.transform(Xd)[0]
    return int(dist.argmax()), dist

def get_topic_keywords_from_bank(bank: dict, topic_id: int, k_each: int = 30) -> Dict[str, List[Tuple[str, float]]]:
    """주제 ID에 해당하는 'good'/'all' 키워드를 (단어, 점수) 튜플 리스트로 반환"""
    if topic_id not in bank or bank[topic_id].get("status") != "ok": # [수정] status check
        return {"good": [], "all": []}
    
    goods = [(w, s) for w,s in bank[topic_id].get("good", [])[:k_each]]
    alls = [(w, s) for w,s in bank[topic_id].get("all", [])[:max(1, k_each//2)]]
    
    seen = set()
    unique_goods = []
    for w,s in goods:
        if w not in seen:
            unique_goods.append((w,s)); seen.add(w)
            
    unique_alls = []
    for w,s in alls:
        if w not in seen:
            unique_alls.append((w,s)); seen.add(w)

    return {"good": unique_goods, "all": unique_alls}


# ====== LLM Reranker ======
def llm_rerank_with_explanations(
    draft_title: str, 
    draft_body: str,  
    candidates: List[str],
    topk: int = 8,
    audience: str = "혼합",
    tone: str = "분석적",
    model_name: str = MODEL_CHAT,
    temperature: float = 0.2,
) -> List[Dict]:
    if not USE_LLM or client is None or not LLM_OK:
        raise RuntimeError("API를 사용할 수 없습니다: OPENAI_API_KEY/네트워크/권한을 확인하세요.")

    cand = [c.strip() for c in candidates if str(c).strip()]
    cand_unique = list(dict.fromkeys(cand))[:500]
    if not cand_unique:
        raise RuntimeError("후보 단어가 비어있습니다. (토픽 단어은행/분류기 힌트 생성 확인)")

    sys_prompt = (
        "너는 한국어 콘텐츠 편집 어시스턴트다. 반드시 JSON 객체만 출력한다. "
        "초안은 {'title': '...', 'body': '...'} JSON 객체로 제공된다. 'title'과 'body'를 명확히 구분하여 분석해야 한다. "
        "객체는 {'items': [...]} 형식이며, 각 항목은 "
        "{term, why, where_to_add, insertion_example, expected_effect, cautions} 키를 가진다. "
        "where_to_add는 반드시 ['제목', '본문'] 중 하나여야 한다. ('소제목', '첫 120자' 등 다른 값은 절대 사용 금지) "
        "반드시 '후보 풀'에 있는 단어만 사용."
        "insertion_example은 반드시 초안의 실제 문장을 찾아 '기존/수정' (Before/After) 형식으로 구체적으로 제시해야 한다."
    )
    user_payload = {
        "goal": f"초안 문맥을 보존하며 후보 풀에서만 Top-{topk} 선별",
        "constraints": [
            "후보 밖 단어/동의어 금지",
            "문맥 어긋나는 삽입 예시 금지",
            "중복 의미 추천 최소화",
            "where_to_add는 '제목' 또는 '본문'만 허용. '소제목' 금지.",
            f"독자수준={audience}",
            f"톤={tone}"
        ],
        "candidates": cand_unique,
        "draft": {
            "title": draft_title,
            "body": draft_body[:6000] 
        },
        "return_format": [
            {"term":"...", "why":"'제목' 또는 '본문'의 문맥과 연관지어 설명", "where_to_add":"제목",
             "insertion_example":"예: '기존: AI 트렌드를 정리합니다.' -> '수정: [추천단어] AI 트렌드를 정리합니다.'", 
             "expected_effect":"...", "cautions":"..."}
        ]
    }
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role":"system","content": sys_prompt},
                {"role":"user","content": json.dumps(user_payload, ensure_ascii=False)}
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        raw = (resp.choices[0].message.content or "").strip()
        st.session_state["llm_meta_last"] = {
            "model": getattr(resp, "model", model_name),
            "id": getattr(resp, "id", ""),
            "usage": getattr(resp, "usage", None),
            "raw": raw,
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        data_obj = _parse_json_safely(raw)
        data = data_obj.get("items") if isinstance(data_obj, dict) else data_obj
        if not isinstance(data, list):
            raise ValueError("JSON 형식 오류: 배열(items)이 아님")

        allowed = set(cand_unique)
        recs: List[Dict] = []
        for item in data:
            term = str(item.get("term","")).strip()
            where = str(item.get("where_to_add","")).strip()
            if not term or term not in allowed or where not in ['제목', '본문']: 
                continue
            recs.append({
                "term": term,
                "category": categorize_term(term),
                "why": str(item.get("why","")).strip(),
                "where_to_add": where,
                "insertion_example": str(item.get("insertion_example","")).strip(),
                "expected_effect": str(item.get("expected_effect","")).strip(),
                "cautions": str(item.get("cautions","")).strip(),
            })
            if len(recs) >= topk:
                break
        if not recs:
            preview = (raw[:200] + "…") if raw else "(빈 응답)"
            raise RuntimeError(f"API 응답 파싱 실패(후보 내 유효 항목 0): {preview}")
        return recs

    except (APIError, RateLimitError) as e:
        raise RuntimeError(f"API 호출 실패: {e}")
    except Exception as e:
        if st.session_state.get("show_llm_raw"):
            st.text(st.session_state.get("llm_meta_last", {}).get("raw","(raw 없음)"))
        raise RuntimeError(f"API 호출/파싱 실패: {e}")

# ========= 성과 기반 분석 (RobustScaler) =========
def build_engagement(df: pd.DataFrame, w_views=0.4, w_likes=0.4, w_comments=0.2) -> pd.DataFrame:
    """
    RobustScaler(중앙값/IQR) 기반 정규화:
      x_rob = (x - median) / IQR
    ※ 범위가 [0,1]로 고정되지 않음(이상치 영향 감소 목적).
    """
    df = df.copy()
    cols = ["views_total", "likes", "comments"]
    for c in cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    scaler = RobustScaler(quantile_range=(25.0, 75.0))
    for c in cols:
        df[c + "_rob"] = scaler.fit_transform(df[[c]]).ravel()

    df["engagement"] = (
        w_views   * df["views_total_rob"] +
        w_likes   * df["likes_rob"] +
        w_comments* df["comments_rob"]
    )
    return df

def label_quality_by_quantile(df: pd.DataFrame, col="engagement", low_q=0.33, high_q=0.66) -> pd.DataFrame:
    df = df.copy()
    q_low, q_high = df[col].quantile([low_q, high_q])
    def _label(x):
        if x >= q_high: return "good"
        if x <= q_low: return "bad"
        return "medium"
    df["quality_label"] = df[col].apply(_label)
    return df

# ========= LDA(온라인) =========
def run_lda_topics_streaming(
    texts: List[str],
    n_topics: int = 10,
    max_features: int = 5000,
    batch_size: int = 1000,
    n_epochs: int = 3,
):
    vect = CountVectorizer(
        min_df=0.01,  # 최소 1%의 문서에는 등장해야 함 (희귀 단어 제거)
        max_df=0.90,   
        stop_words=STOPWORDS_KO
    )
    X = vect.fit_transform([t if isinstance(t, str) else "" for t in texts])

    lda = LatentDirichletAllocation(
        n_components=n_topics, learning_method="online",
        batch_size=batch_size, max_iter=1, random_state=42, evaluate_every=0,
    )

    n_samples = X.shape[0]
    n_batches = int(np.ceil(n_samples / batch_size))
    total_steps = n_epochs * n_batches

    prog = st.progress(0.0, text="LDA 주제 분석 학습 중…")
    t0 = time.time(); step = 0

    idx_all = np.arange(n_samples)
    for epoch in range(n_epochs):
        idx_all = sk_shuffle(idx_all, random_state=42 + epoch)
        for b in range(n_batches):
            bs = idx_all[b * batch_size : (b + 1) * batch_size]
            Xb = X[bs]
            lda.partial_fit(Xb)

            step += 1
            frac = step / total_steps
            elapsed = time.time() - t0
            sec_per_step = elapsed / max(step, 1)
            remain = sec_per_step * (total_steps - step)
            prog.progress(
                frac, text=f"LDA 학습 {frac*100:.1f}% | 경과 {elapsed:,.0f}s | 남음 ~{remain:,.0f}s"
            )

    W = lda.transform(X)        # 문서-토픽 분포
    prog.empty()
    df_topic = pd.DataFrame({"topic": W.argmax(axis=1)})
    return df_topic, vect, lda, W

# --------- n-gram 중요도 ---------
def top_ngrams_by_logreg(texts: List[str], labels: List[str], ngram_range=(1,2), k: int = 20) -> Dict[str, List[Tuple[str, float]]]:
    y = np.array([1 if l=="good" else 0 for l in labels])
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range, 
        max_features=30000, 
        min_df=2,
        stop_words=STOPWORDS_KO 
    )
    X = tfidf.fit_transform([t if isinstance(t,str) else "" for t in texts])
    clf = LogisticRegression(max_iter=1000, solver="liblinear")
    clf.fit(X, y)
    coefs = clf.coef_[0]
    idx_sorted = np.argsort(coefs)
    vocab = np.array(tfidf.get_feature_names_out())
    top_pos = [(vocab[i], float(coefs[i])) for i in idx_sorted[::-1][:k]]
    top_neg = [(vocab[i], float(coefs[i])) for i in idx_sorted[:k]]
    return {"good_terms": top_pos, "bad_terms": top_neg}

# --------- 불용어/ETA 학습 유틸 ---------
def train_logreg_with_progress(texts, labels, stoplist=None, ngram_range=(1,2),
                               epochs=3, batch_size=2000, k_show=20, seed=42):
    y = np.array([1 if l=="good" else 0 for l in labels])
    
    final_stop_words = list(STOPWORDS_KO)
    if stoplist:
        if isinstance(stoplist, (set, tuple, list)):
            final_stop_words.extend(list(stoplist))
        else:
            try: final_stop_words.extend(list(stoplist))
            except Exception: pass
    final_stop_words = list(set(final_stop_words)) 

    tfidf = TfidfVectorizer(
        ngram_range=ngram_range, max_features=30000, min_df=2,
        stop_words=final_stop_words 
    )
    X = tfidf.fit_transform([t if isinstance(t,str) else "" for t in texts])

    n = X.shape[0]
    idx_all = np.arange(n)
    n_batches = int(np.ceil(n / batch_size))
    total_steps = epochs * n_batches

    clf = SGDClassifier(loss="log_loss", learning_rate="optimal",
                        alpha=1e-5, random_state=seed, warm_start=True)
    classes = np.array([0,1])

    prog = st.progress(0.0, text="콘텐츠 등급 예측 모델 학습 중…")
    t0 = time.time(); step = 0

    for ep in range(epochs):
        idx_all = sk_shuffle(idx_all, random_state=seed + ep)
        for b in range(n_batches):
            bs = idx_all[b*batch_size : (b+1)*batch_size]
            Xb = X[bs]; yb = y[bs]
            clf.partial_fit(Xb, yb, classes=classes)

            step += 1
            frac = step/total_steps
            elapsed = time.time() - t0
            sec_per = elapsed / max(step,1)
            remain = sec_per * (total_steps - step)
            prog.progress(frac, text=f"예측 모델 학습 {frac*100:.1f}% | 경과 {elapsed:,.0f}s | 남음 ~{remain:,.0f}s")

    prog.empty()

    coefs = clf.coef_[0]
    vocab = np.array(tfidf.get_feature_names_out())
    order = np.argsort(coefs)
    good_terms = [(vocab[i], float(coefs[i])) for i in order[::-1][:k_show]]
    bad_terms  = [(vocab[i], float(coefs[i]))  for i in order[:k_show]]

    return {
        "tfidf": tfidf, "clf": clf,
        "good_terms": good_terms, "bad_terms": bad_terms,
        "last_eta_info": {"elapsed": time.time()-t0}
    }

# === 토픽 상위단어 & 라벨 ===
def get_topic_top_words(lda, vect, topn=8):
    vocab = np.array(vect.get_feature_names_out())
    topics = {}
    for k, comp in enumerate(lda.components_):
        idx = np.argsort(comp)[::-1][:topn]
        topics[f"Topic {k}"] = [str(vocab[i]) for i in idx]
    return topics

def _heuristic_topic_name(words: list[str]) -> dict:
    w = " ".join(words)
    rules = [
        (["정부","국회","예산","정책","대통령"], ("정치/행정", "정부·국회·예산 관련 이슈")),
        (["손흥민","리그","경기","골","선수","스포츠"], ("스포츠/축구", "경기/선수/리그 중심 기사")),
        (["AI","인공지능","로봇","기술","산업","자동화","데이터"], ("기술/AI", "AI·로봇·산업 자동화")),
        (["주식","환율","부동산","금리","경제"], ("경제/금융", "거시경제·시장 동향")),
        (["코로나","의료","건강","병원"], ("의료/건강", "질병·의료·헬스케어")),
        (["넷플릭스","영화","드라마","콘텐츠"], ("문화/콘텐츠", "영화·방송·플랫폼")),
    ]
    for keys, (nm, desc) in rules:
        if any(k in w for k in keys):
            return {"name": nm, "desc": desc}
    return {"name": "일반/종합", "desc": "광범위한 이슈 묶음"}

def llm_name_topics(topic_top_words: dict, model_name=MODEL_CHAT):
    if not USE_LLM or client is None or not LLM_OK:
        return {k: _heuristic_topic_name(v) for k, v in topic_top_words.items()}

    payload = {
        "topics": topic_top_words,
        "schema": {"Topic k": {"name": "짧은 이름", "desc": "한 줄 설명"}},
        "instruction": "위 'topics'의 상위 단어를 보고 각 토픽에 대해 {name, desc}를 생성. "
                       "JSON 객체로 { 'Topic 0': {'name':'..','desc':'..'}, ... } 형식만 반환. 다른 텍스트 금지."
    }
    try:
        r = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role":"system","content":"너는 주제 라벨러다. JSON 객체만 반환한다."},
                {"role":"user","content": json.dumps(payload, ensure_ascii=False)}
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        txt = (r.choices[0].message.content or "").strip()
        data = _parse_json_safely(txt)
        if not isinstance(data, dict) or not data:
            raise ValueError("빈 JSON")
        for k, words in topic_top_words.items():
            if k not in data or "name" not in data[k]:
                data[k] = _heuristic_topic_name(words)
        return data
    except Exception:
        return {k: _heuristic_topic_name(v) for k, v in topic_top_words.items()}

# [수정] 감성 S/I 계산 함수 시그니처 변경 (cv, lex를 인자로 받도록)
def compute_sentiment_SI(df_work: pd.DataFrame, cv: CountVectorizer, lex: dict) -> pd.DataFrame:
    """간단 토큰 기준 평균감성 S, 평균절대감성 I (CV, Lexicon 외부 주입)"""
    df = df_work.copy()
    texts = (df["title"].fillna("") + " " + df["content"].fillna("")).tolist()

    X = cv.transform(texts) 
    vocab = np.array(cv.get_feature_names_out())

    rows, cols = X.nonzero()
    tok_by_row: Dict[int, List[str]] = {}
    for r, c in zip(rows, cols):
        tok_by_row.setdefault(r, []).append(vocab[c])

    S, I = [], []
    for r in range(X.shape[0]):
        vals = [lex.get(t, 0.0) for t in tok_by_row.get(r, [])]
        if vals:
            S.append(float(np.mean(vals)))
            I.append(float(np.mean(np.abs(vals))))
        else:
            S.append(0.0); I.append(0.0)
    df["S"], df["I"] = S, I
    return df

# [수정] TAB1에서 초안 텍스트의 감성 점수를 계산하기 위한 헬퍼 함수
def get_sentiment_for_text(txt: str, senti_pack: dict) -> Tuple[float, float]:
    """단일 텍스트에 대해 S/I 점수 계산"""
    if not senti_pack or not senti_pack.get('cv') or not senti_pack.get('lex') or not txt:
        return 0.0, 0.0
    
    try:
        cv = senti_pack['cv']
        lex = senti_pack['lex']
        
        X = cv.transform([txt])
        vocab = np.array(cv.get_feature_names_out())
        
        rows, cols = X.nonzero()
        if not np.any(cols): 
            return 0.0, 0.0
            
        vals = [lex.get(vocab[c], 0.0) for c in cols]
        if vals:
            s = float(np.mean(vals))
            i = float(np.mean(np.abs(vals)))
            return s, i
    except Exception:
        return 0.0, 0.0
    return 0.0, 0.0

# [신규] TAB1에서 최근 30일 인기 단어를 동적으로 추출하는 함수
def get_recent_popular_words(df_all_data: pd.DataFrame, 
                             end_date: datetime.date, 
                             topic_id: int = None, 
                             k: int = 10) -> List[str]:
    """특정 토픽/기간/Good등급 문서에서 Top-K 빈도 단어 추출"""
    if df_all_data is None or df_all_data.empty or 'date' not in df_all_data.columns or 'topic' not in df_all_data.columns or 'quality_label' not in df_all_data.columns:
        return []
    
    try:
        df = df_all_data.copy()
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
             df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # 날짜 필터링
        end_date_pd = pd.to_datetime(end_date)
        start_date_pd = end_date_pd - pd.Timedelta(days=30)
        
        df_filtered = df[
            (df['date'] >= start_date_pd) & 
            (df['date'] <= end_date_pd) &
            (df['quality_label'] == 'good')
        ]
        
        if topic_id is not None:
            df_filtered = df_filtered[df_filtered['topic'] == topic_id]
        
        if df_filtered.empty:
            return []
        
        texts = (df_filtered["title"].fillna("") + " " + df_filtered["content"].fillna("")).tolist()
        
        cv_recent = CountVectorizer(max_features=2000, stop_words=STOPWORDS_KO)
        X_recent = cv_recent.fit_transform(texts)
        
        word_counts = X_recent.sum(axis=0)
        words_freq = [(word, word_counts[0, idx]) for word, idx in cv_recent.vocabulary_.items()]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        
        return [word for word, freq in words_freq[:k]]
        
    except Exception as e:
        st.error(f"[최근 단어 분석 오류] {e}. (데이터에 'topic' 또는 'date' 컬럼이 올바르게 포함되었는지 확인하세요.)")
        return []


# ========= 회귀 유틸 =========
def fit_ols(y, X):
    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
    y = y.apply(pd.to_numeric, errors='coerce').fillna(0)
    
    valid_idx = (y != 0) | (X != 0).any(axis=1)
    y_valid = y[valid_idx]
    X_valid = X[valid_idx]

    if len(y_valid) < 2:
        raise ValueError("유효한 데이터 포인트가 2개 미만입니다. 회귀 분석을 실행할 수 없습니다.")
        
    Xc = sm.add_constant(X_valid, has_constant="add")
    model = sm.OLS(y_valid.astype(float), Xc, missing="drop")
    return model.fit()

def tidy_summary(res: sm.regression.linear_model.RegressionResultsWrapper, max_rows=200):
    s = []
    for name, coef, se, t, p in zip(res.params.index, res.params.values, res.bse.values, res.tvalues, res.pvalues):
        s.append({"term": name, "coef": float(coef), "se": float(se), "t": float(t), "p": float(p)})
    df = pd.DataFrame(s)
    if len(df) > max_rows:
        return df.head(max_rows)
    return df

# [수정] TAB4 '불용어 의심 단어' 로직 변경: 'Bad'가 아닌 'Generic' 단어 추출
def get_suspected_stopwords(df_all_data: pd.DataFrame, k: int = 50) -> List[str]:
    """토픽/성과와 무관하게 가장 자주 쓰이는 일반 단어(불용어 후보) 추출"""
    if df_all_data is None or df_all_data.empty:
        return []
    try:
        texts = (df_all_data["title"].fillna("") + " " + df_all_data["content"].fillna("")).tolist()
        
        # [수정] 불용어 *없이* CV 실행, min_df=0.1 (최소 10%의 문서에 등장)
        cv_nostop = CountVectorizer(max_features=k, 
                                    min_df=0.1, 
                                    ngram_range=(1,1)) 
        cv_nostop.fit(texts)
        common_words = cv_nostop.get_feature_names_out()
        
        # 이 중에서 이미 우리 기본 불용어 리스트에 있는 것은 제외
        final_suspects = [w for w in common_words if w not in STOPWORDS_KO]
        return final_suspects
    except Exception as e:
        st.error(f"[불용어 의심 단어 추출 오류] {e}")
        return []

# [신규] TAB1 감성 게이지 차트 생성 함수
def create_sentiment_gauge(s_val: float, s_target: float, i_val: float, i_target: float, lexicon_max: float = 1.0):
    """Plotly의 Bullet Chart를 사용해 감성 점수(S)와 강도(I) 게이지 생성"""
    
    fig = go.Figure()

    # [수정] 게이지를 가로로 배치 (domain x, y값 수정)
    # 1. 감성 점수 (S) 게이지 (왼쪽)
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = s_val,
        domain = {'x': [0, 0.48], 'y': [0, 1]}, # [수정]
        title = {'text': "💖 감성 점수 (S)", 'font': {'size': 18}},
        number = {'font': {'size': 24}},
        gauge = {
            'axis': {'range': [-lexicon_max, lexicon_max], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#0072F0" if s_val >= 0 else "#E63946", 'thickness': 0.4},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#CCCCCC",
            'steps': [
                {'range': [-lexicon_max, -0.05], 'color': 'rgba(230, 57, 70, 0.1)'}, 
                {'range': [-0.05, 0.05], 'color': 'rgba(200, 200, 200, 0.2)'}, 
                {'range': [0.05, lexicon_max], 'color': 'rgba(0, 114, 240, 0.1)'}
            ],
            'threshold': {
                'line': {'color': "green", 'width': 3},
                'thickness': 0.75,
                'value': s_target
            }
        }
    ))

    # 2. 감성 강도 (I) 게이지 (오른쪽)
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = i_val,
        domain = {'x': [0.52, 1], 'y': [0, 1]}, # [수정]
        title = {'text': "💖 감성 강도 (I)", 'font': {'size': 18}},
        number = {'font': {'size': 24}},
        gauge = {
            'axis': {'range': [0, lexicon_max], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#F4A261", 'thickness': 0.4},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#CCCCCC",
            'steps': [
                {'range': [0, lexicon_max * 0.33], 'color': 'rgba(200, 200, 200, 0.2)'},
                {'range': [lexicon_max * 0.33, lexicon_max * 0.66], 'color': 'rgba(244, 162, 97, 0.1)'},
                {'range': [lexicon_max * 0.66, lexicon_max], 'color': 'rgba(244, 162, 97, 0.2)'},
            ],
            'threshold': {
                'line': {'color': "green", 'width': 3},
                'thickness': 0.75,
                'value': i_target
            }
        }
    ))

    fig.update_layout(
        height=180, # [수정] 세로 크기 조정
        margin=dict(l=20, r=20, t=40, b=10)
    )
    return fig


# ==================== (★) MODE_CFG ====================
MODE_CFG = {
    "quick": {
        "sample_n": 5000,
        "lda_topics": 0,
        "max_features": 3000,
        "batch_size": 500,
        "n_epochs": 2,
        "clf_epochs": 1,
        "clf_batch": 500,
        "ngram_range": (1, 2),
    },
    "full": {
        "sample_n": None,
        "lda_topics": 0,
        "max_features": 5000,
        "batch_size": 1000,
        "n_epochs": 3,
        "clf_epochs": 3,
        "clf_batch": 1000,
        "ngram_range": (1, 3),
    },
}
# ==================================================================

# ========= Streamlit UI =========
st.set_page_config(page_title="문맥형 추천 + 성과 분석 + 감성/회귀", page_icon="📝", layout="wide")
st.title("📝 문맥형 용어 추천 + 📈 성과 분석 대시보드")

with st.sidebar:
    st.subheader("공통 설정")
    audience = st.selectbox("주요 독자 수준", ["입문자", "전문가", "혼합"], index=2)
    tone = st.selectbox("콘텐츠 톤/스타일", ["친근", "공식", "분석적"], index=2)
    if LLM_OK: st.success("LLM 상태: ✅ 연결 OK")
    elif USE_LLM: st.error("LLM 상태: ❌ 인증/권한/네트워크 오류")
    else: st.error("LLM 상태: ❌ OPENAI_API_KEY 미설정")

def require_llm():
    if not LLM_OK:
        st.error("API를 사용할 수 없습니다: OPENAI_API_KEY/네트워크/권한을 확인하세요.")
        st.stop()

# [신규] TAB4 추가
TAB1, TAB2, TAB3, TAB4 = st.tabs([
    "💡 문맥형 용어 추천", 
    "📈 성과/주제/감성 분석", 
    "🛠️ 파인튜닝(골격)",
    "🔬 모델 관리자 (Admin)" 
])

# ================= TAB1 =================
with TAB1:
    st.header("1) 초안 텍스트 입력")
    
    draft_title = st.text_input("제목 (선택)", placeholder="예: 이번 주 AI 트렌드 Top 5")
    draft_body = st.text_area("본문 (초안)", height=220,
                         placeholder="예) 1. 오픈AI의 새 모델이...")
    
    full_draft = draft_title.strip() + " " + draft_body.strip()

    c_date, c_check = st.columns([1, 1])
    with c_date:
        ref_date = st.date_input("기준 날짜", datetime.date.today())
    with c_check:
        st.write("") 
        st.write("")
        all_dates = st.checkbox("모든 날짜 선택하기 (전체 기간 분석)", value=True) 

    # [수정] 'LLM 토큰 보기' 제거

    for _k, _v in [
        ("last_recs", None),
        ("last_recs_time", None),
        ("last_draft", ""),
        ("last_candidates", []),
        ("sentiment_pack", None), 
        ("df_for_analysis", None),
        ("analysis_done", False), 
    ]:
        st.session_state.setdefault(_k, _v)

    candidates = list(DEFAULT_CANDIDATES)
    topic_id_for_draft, topic_dist = None, None
    topic_name = "미분류" 

    topic_bank = st.session_state.get("topic_term_bank")
    lda_vect   = st.session_state.get("lda_vect")
    lda_model  = st.session_state.get("lda_model")
    clf_pack   = st.session_state.get("clf_pack")
    senti_pack = st.session_state.get('sentiment_pack')
    df_all_data = st.session_state.get('df_for_analysis') 

    # [신규] 데이터 미로드 시 경고
    if not st.session_state['analysis_done']:
        st.warning("⚠️ **데이터 미로드:** TAB2에서 CSV 업로드 및 분석을 실행하면 과거 데이터 기반의 확률, 주제, 키워드 추천이 활성화됩니다.")

    
    # --- 1. 토픽 추론 및 태그 표시 ---
    if full_draft.strip() and topic_bank and lda_vect is not None and lda_model is not None:
        topic_id_for_draft, topic_dist = infer_topic_for_text(full_draft, lda_vect, lda_model)
        
        topic_name = f"토픽 {topic_id_for_draft}"
        topic_desc = "분석된 주제"
        lbls = st.session_state.get("topic_labels", {})
        if f"Topic {topic_id_for_draft}" in lbls:
            meta = lbls[f"Topic {topic_id_for_draft}"]
            topic_name = meta.get('name', topic_name)
            topic_desc = meta.get('desc', topic_desc)

        st.markdown(f"**초안의 예상 주제:** <span style='background-color: #0072F0; color: white; padding: 3px 8px; border-radius: 15px; font-size: 0.9em; margin-left: 10px;'>{topic_name}</span>", unsafe_allow_html=True)
        st.caption(f"└ {topic_desc} (후보 단어 {len(candidates)}개)")
    
    # --- 2. 등급/확률 및 키워드 추천 (분류기 로드 시) ---
    if full_draft.strip() and clf_pack is not None:
        tfidf = clf_pack["tfidf"]; clf = clf_pack["clf"]
        Xd = tfidf.transform([full_draft])
        proba_good = float(clf.predict_proba(Xd)[0,1])
        label = "상 (Good)" if proba_good >= 0.5 else "하 (Bad)"
        
        c1, c2 = st.columns(2)
        c1.metric("예상 콘텐츠 매력 등급", label)
        c2.metric("📈 과거 데이터 기반 성과 확률", f"{proba_good*100:.1f}%")
        
        st.caption("└ 과거 데이터(TAB2)로 학습한 통계 모델의 예측치입니다. LLM 추천(문맥/품질 중심)과 다를 수 있습니다.")

        # [수정] '모든 날짜' 체크박스에 따라 동적 키워드 추천
        if all_dates:
            # --- "모든 날짜" 선택 시 (기존 로직) ---
            if topic_id_for_draft is not None and topic_bank:
                topic_keywords_data = get_topic_keywords_from_bank(topic_bank, int(topic_id_for_draft), k_each=10)
                good_topic_terms = [w for w,s in topic_keywords_data.get("good", [])]
                if good_topic_terms:
                    with st.expander(f"✅ **'{topic_name}' 주제**의 **전체 기간** 성과 우수 단어 (추천)"):
                        st.markdown(f"**이유:** 과거 이 주제(`{topic_name}`)의 콘텐츠 중 **높은 성과**를 낸 문서에서 자주 발견된 단어들입니다.")
                        st.info(", ".join(good_topic_terms))
                    candidates = list(dict.fromkeys(good_topic_terms + candidates)) 

            good_terms_list = clf_pack.get("good_terms", [])
            top_good = [t for t,_ in good_terms_list]
            missing  = [t for t in top_good if t not in full_draft][:8]
            if missing:
                st.info(f"💡 **전체 기간** 성과 우수 단어 (초안에 없으면 추가 고려): \n\n" + ", ".join(missing))
                st.caption("└ 이유: 주제와 관계없이 전반적으로 높은 성과를 낸 콘텐츠의 공통 키워드입니다.")
                candidates = list(dict.fromkeys(list(missing) + candidates))
        
        else:
            # --- "특정 날짜" 선택 시 (최근 30일) ---
            if topic_id_for_draft is not None and (df_all_data is not None and not df_all_data.empty):
                # 1. 주제별 최근 단어
                recent_topic_words = get_recent_popular_words(df_all_data, ref_date, topic_id=topic_id_for_draft, k=10)
                if recent_topic_words:
                    with st.expander(f"📈 **'{topic_name}' 주제**의 **최근 30일** 인기 단어 (Good 콘텐츠)"):
                        st.markdown(f"**기준:** `{ref_date - datetime.timedelta(days=30)}` ~ `{ref_date}` 기간 동안 성과가 좋았던 문서 기준")
                        st.warning(", ".join(recent_topic_words))
                    candidates = list(dict.fromkeys(recent_topic_words + candidates))
                
                # 2. 전체 최근 단어
                recent_all_words = get_recent_popular_words(df_all_data, ref_date, topic_id=None, k=10)
                missing_recent = [w for w in recent_all_words if w not in full_draft][:8]
                if missing_recent:
                    st.info(f"📈 **전체 주제**의 **최근 30일** 인기 단어 (초안에 없으면 추가 고려): \n\n" + ", ".join(missing_recent))
                    st.caption(f"└ 이유: `{ref_date}` 기준 최근 30일간 성과가 좋았던 모든 콘텐츠의 공통 키워드입니다.")
                    candidates = list(dict.fromkeys(list(missing_recent) + candidates))
                
                if not recent_topic_words and not missing_recent:
                    st.caption(f"ℹ️ `{ref_date}` 기준 최근 30일간의 인기 단어 정보가 없습니다.")
            elif df_all_data is None or df_all_data.empty:
                 st.caption("ℹ️ TAB2에서 분석을 실행하면 '최근 30일 인기 단어'가 활성화됩니다.")

        st.divider()
        
        # --- 3. 감성 점수 (감성 사전 로드 시) ---
        if senti_pack and senti_pack.get('cv') and senti_pack.get('lex'):
             senti_s, senti_i = get_sentiment_for_text(full_draft, senti_pack)
             target_s = senti_pack.get('target_s')
             target_i = senti_pack.get('target_i')

             # [신규] Plotly 게이지 차트로 S/I 점수와 목표치 표시
             st.plotly_chart(
                 create_sentiment_gauge(senti_s, target_s, senti_i, target_i, lexicon_max=1.0),
                 use_container_width=True
             )
             
             # [신규] 캡션에 목표 점수 텍스트 추가
             if target_s is not None and target_i is not None:
                st.markdown(f"**🎯 목표 점수** (Good 콘텐츠 평균): **S (점수): {target_s:.2f}** | **I (강도): {target_i:.2f}**")
             
             st.caption("""
             * **감성 점수 (S):** 긍정(>0) 또는 부정(<0)의 정도. 0은 중립. (범위: -1 ~ 1)
             * **감성 강도 (I):** 감성이 얼마나 강하게 표현되었는지. 0은 감성 단어 없음. (범위: 0 ~ 1)
             * **초록색 선:** 성과가 좋았던(Good) 글들의 평균 '목표' 점수입니다.
             * **가정:** 이 차트는 사용된 감성 사전의 점수가 -1에서 +1 사이임을 가정합니다.
             """)
             
             # [신규] 동적 감성 추천
             if target_s is not None and target_i is not None:
                recs = []
                if abs(senti_s - target_s) > 0.05:
                    if senti_s < target_s:
                        recs.append(f"성과 좋은 글(목표 {target_s:.2f})은 **더 긍정적**입니다. (긍정 단어 추가)")
                    else:
                        recs.append(f"성과 좋은 글(목표 {target_s:.2f})은 **더 차분(중립/부정)**합니다. (긍정 단어 감소)")
                if abs(senti_i - target_i) > 0.1:
                    if senti_i < target_i:
                        recs.append(f"성과 좋은 글(목표 {target_i:.2f})은 **감성 표현이 더 강합니다.** (감성 단어 추가)")
                    else:
                        recs.append(f"성과 좋은 글(목표 {target_i:.2f})은 **더 객관적(감성 강도 낮음)**입니다. (감성 단어 감소)")
                
                if recs:
                    st.info("💡 **[감성 추천]** " + " ".join(recs))

        else:
             st.info("💖 감성 점수 (S/I)\n\nTAB2에서 '감성 사전'을 업로드하고 분석을 실행하면 활성화됩니다.")
        
        st.divider()

    # --- 4. LLM 리랭커 ---
    st.subheader("2) 문맥형 추천 (LLM 리랭커, 기존 후보 단어만 사용)")
    topk = st.slider("추천 개수 (Top-K)", 3, 15, 8)
    btn = st.button("✨ 문맥형 용어 추천 생성", disabled=not LLM_OK)

    if btn:
        require_llm()
        if not full_draft.strip():
            st.warning("제목이나 본문 초안 텍스트를 입력하세요.")
        else:
            with st.spinner("LLM이 초안 문맥에 맞는 단어를 선별 중입니다..."):
                try:
                    recs = llm_rerank_with_explanations(
                        draft_title=draft_title, 
                        draft_body=draft_body, 
                        candidates=candidates, 
                        topk=topk, 
                        audience=audience, 
                        tone=tone
                    )
                    st.success("추천 완료!")
                    st.session_state["last_recs"] = recs
                    st.session_state["last_recs_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    st.session_state["last_draft"] = full_draft
                    st.session_state["last_candidates"] = list(candidates)
                except Exception as e:
                    st.error(str(e))

    if st.session_state.get("last_recs"):
        for i, r in enumerate(st.session_state["last_recs"], 1):
            with st.expander(f"**{i}. {r.get('term','(용어)')}** — 분류: {r.get('category','')}", expanded=(i==1)):
                st.markdown(f"**추천 이유**: {r.get('why','')}") 
                st.markdown(f"**어디에 삽입할까요?**: **{r.get('where_to_add','')}**") 
                if r.get("insertion_example"): st.markdown(f"**예시**: {r['insertion_example']}")
                if r.get("expected_effect"):   st.markdown(f"**예상되는 효과**: {r['expected_effect']}") 
                if r.get("cautions"):          st.markdown(f"**사용 시 주의할 점**: {r['cautions']}")
    else:
        st.info("아직 생성된 추천이 없습니다. 위 버튼으로 먼저 생성하세요.")

    st.markdown("---")
    st.caption("• 추천 용어는 TAB2에서 분석한 데이터 기반으로 생성된 '토픽별 핵심 단어' 풀에서 선별됩니다.")

# ================= TAB2 =================
with TAB2:
    st.header("📊 데이터 업로드 및 성과 분석")
    st.caption("콘텐츠 성과를 분석하려면 아래 두 가지 CSV 파일이 모두 필요합니다.")
    c1, c2 = st.columns(2)
    with c1:
        f_content = st.file_uploader("📝 (1) 콘텐츠 상세 CSV (article_id, title, content, date 포함)", type=["csv"], key="content")
    with c2:
        f_metrics = st.file_uploader("📈 (2) 성과 측정 CSV (article_id, views_total, likes, comments, period 포함)", type=["csv"], key="metrics")

    st.markdown("---")
    st.subheader("⚙️ 분석 설정")
    c3, c4, c5 = st.columns(3)
    lda_topics = c3.number_input("주제 분류 개수 (LDA 토픽 수)", min_value=5, max_value=40, value=10, step=1)
    c4.markdown("**콘텐츠 매력 점수 가중치** (총합 1.0)")
    wv = c4.slider("조회수 가중치", 0.0, 1.0, 0.4, 0.05, key="wv_slider")
    wl = c4.slider("좋아요 가중치", 0.0, 1.0, 0.4, 0.05, key="wl_slider")
    wc = c4.slider("댓글 가중치", 0.0, 1.0, 0.2, 0.05, key="wc_slider")
    
    f_lex = c5.file_uploader("💖 감성 사전 CSV (선택: word,score)", type=["csv"], key="lex")

    st.session_state.setdefault('analysis_done', False)

    def prepare_by_mode(df_in: pd.DataFrame, mode_cfg: dict, lda_topics_ui: int):
        if mode_cfg["sample_n"]:
            n = min(mode_cfg["sample_n"], len(df_in))
            df_work = df_in.sample(n=n, random_state=42).reset_index(drop=True)
            st.info(f"샘플링 사용: **{n}행**으로 축소(원본 {len(df_in)}).")
        else:
            df_work = df_in.copy()
        
        n_topics = mode_cfg["lda_topics"] if mode_cfg["lda_topics"] > 0 else int(lda_topics_ui) 

        lda_kwargs = dict(
            n_topics=n_topics,
            max_features=mode_cfg["max_features"],
            batch_size=mode_cfg["batch_size"],
            n_epochs=mode_cfg["n_epochs"],
        )
        clf_kwargs = dict(
            epochs=mode_cfg["clf_epochs"],
            batch_size=mode_cfg["clf_batch"],
            ngram_range=mode_cfg["ngram_range"],
        )
        return df_work, lda_kwargs, clf_kwargs

    @st.cache_resource(show_spinner=False)
    def cached_lda_run(texts_tuple, n_topics, max_features, batch_size, n_epochs):
        return run_lda_topics_streaming(
            list(texts_tuple), n_topics=n_topics, max_features=max_features,
            batch_size=batch_size, n_epochs=n_epochs
        )

    if f_content is not None and f_metrics is not None:
        try:
            if (st.session_state.get('f_content_name') != f_content.name) or \
               (st.session_state.get('f_metrics_name') != f_metrics.name):
                st.session_state['analysis_done'] = False
                st.session_state['f_content_name'] = f_content.name
                st.session_state['f_metrics_name'] = f_metrics.name
                st.info("새로운 파일이 감지되었습니다. 분석을 다시 실행해주세요.")

            df_c = coerce_article_id(read_csv_robust(f_content))
            df_m_raw = coerce_article_id(read_csv_robust(f_metrics)) 
            df_c["article_id"] = df_c["article_id"].astype(str)
            df_m_raw["article_id"] = df_m_raw["article_id"].astype(str)

            st.info("성과 CSV(2)가 기간별 데이터(long format)입니다. article_id 기준으로 성과(views, likes, comments)를 **합산(sum)**합니다.")
            metric_cols = ["views_total", "likes", "comments"]
            
            for col in metric_cols:
                if col in df_m_raw.columns:
                    df_m_raw[col] = pd.to_numeric(df_m_raw[col], errors='coerce').fillna(0)
            
            df_m = df_m_raw.groupby("article_id")[metric_cols].sum().reset_index()

            df = pd.merge(df_c, df_m, on="article_id", how="inner")
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            st.session_state['df_for_analysis'] = df.copy() 
            
            st.success(f"데이터 병합 완료: **{len(df)}** 건의 **고유 콘텐츠** 분석 준비 완료 (성과는 Total Sum 기준)")

            df = build_engagement(df, w_views=wv, w_likes=wl, w_comments=wc)
            df = label_quality_by_quantile(df, col="engagement", low_q=0.33, high_q=0.66)
            
            st.session_state['df_for_analysis']['quality_label'] = df['quality_label'].values

            st.subheader("1. 콘텐츠 등급 확인")
            st.caption("콘텐츠 매력 점수(Total Engagement)를 기준으로 상위 33%는 '상 (good)', 하위 33%는 '하 (bad)'로 분류했습니다.")
            grade_counts = df["quality_label"].value_counts().rename({"good": "상 (Good)", "medium": "중 (Medium)", "bad": "하 (Bad)"})
            st.dataframe(grade_counts.to_frame(name="콘텐츠 수"), use_container_width=True)


            colm1, colm2 = st.columns(2)
            do_quick = colm1.button("⚡️ 빠른 분석 (샘플/경량 모델)", use_container_width=True)
            do_full = colm2.button("🔬 정밀 분석 (전체/고정밀 모델)", use_container_width=True)

            if do_quick or do_full:
                # ----------------- 분석 실행 블록 시작 -----------------
                mode = "quick" if do_quick else "full"
                cfg = MODE_CFG[mode] 
                df_work, lda_kw, clf_kw = prepare_by_mode(df, cfg, lda_topics)

                # ===== LDA =====
                st.subheader("2. 주제(토픽) 분류 및 분석")
                with st.spinner(f"LDA({mode}) 주제 분석 실행 중… (불용어 적용)"):
                    df_sig = tuple(df_work["content"].fillna("").tolist())
                    df_topic, vect, lda, W = cached_lda_run(tuple(df_sig), **lda_kw)
                df_work["topic"] = df_topic["topic"]
                st.write("주제 분류 결과 (샘플):", df_work[["article_id","topic","title"]].head(10))

                topics_top_words = get_topic_top_words(lda, vect, topn=8)
                st.info("토픽 키워드를 기반으로 LLM/휴리스틱을 사용해 주제 이름과 설명을 추론합니다.")
                with st.spinner("LLM/휴리스틱으로 토픽 라벨링 중..."):
                    topic_labels = llm_name_topics(topics_top_words)

                st.session_state["topic_labels"] = topic_labels
                st.session_state["lda_vect"] = vect 
                st.session_state["lda_model"] = lda
                
                if 'df_for_analysis' in st.session_state and st.session_state['df_for_analysis'] is not None:
                    full_texts = (st.session_state['df_for_analysis']["title"].fillna("") + " " + st.session_state['df_for_analysis']["content"].fillna("")).tolist()
                    full_X = vect.transform(full_texts)
                    full_topics = lda.transform(full_X).argmax(axis=1)
                    st.session_state['df_for_analysis']['topic'] = full_topics
                    st.info("전체 데이터(TAB1/TAB4용)에 토픽 분류 적용 완료.")

                # ===== 분류기 학습 =====
                st.subheader("3. 콘텐츠 등급 예측 모델 학습")
                df_train = df_work[df_work["quality_label"] != "medium"]
                st.info(f"학습 데이터셋: '상(Good)'과 '하(Bad)' {len(df_train)}건 사용.")
                
                with st.spinner(f"SGD 분류기 학습 중… (불용어 적용)"):
                    clf_pack = train_logreg_with_progress(
                        texts = df_train["title"].fillna("") + " " + df_train["content"].fillna(""),
                        labels = df_train["quality_label"],
                        stoplist=None, 
                        **clf_kw
                    )
                st.session_state["clf_pack"] = clf_pack
                st.success("등급 예측 모델 학습 완료 및 키워드 추출 완료!")
                
                # ===== [수정] 토픽 단어 은행 구축 (Chi2) =====
                st.subheader("4. 토픽별 핵심 단어 은행 구축 (TAB1 추천 기반)")
                with st.spinner("토픽별 성과 우수/저조 단어 분석 중… (Chi2 통계 적용)"):
                    df_full_with_topic = st.session_state.get('df_for_analysis', df_work) 
                    # [수정] Chi2 함수 호출, lda_vect 전달
                    topic_term_bank = build_topic_term_bank(df_full_with_topic, vect, lda, topn=50) 
                    st.session_state["topic_term_bank"] = topic_term_bank
                    st.success("토픽 기반 용어 은행(Chi2) 구축 완료! (TAB1에서 활용 가능)")

                # [수정] 감성 분석기 생성 로직 (TAB1용)
                if f_lex is not None:
                    st.subheader("5. 감성 분석기 생성 (TAB1용)")
                    with st.spinner("감성 사전을 처리하여 TAB1에서 사용할 분석기를 생성 중입니다..."):
                        try:
                            f_lex.seek(0)
                            lex_df = read_csv_robust(f_lex)
                            if not set(["word","score"]).issubset(lex_df.columns):
                                st.warning("감성 사전에 'word', 'score' 컬럼이 없어 S/I 계산을 건너뜁니다.")
                                st.session_state['sentiment_pack'] = None
                            else:
                                lex_dict = dict(zip(lex_df["word"].astype(str), lex_df["score"].astype(float)))
                                senti_cv = CountVectorizer(min_df=1, stop_words=STOPWORDS_KO)
                                texts = (df["title"].fillna("") + " " + df["content"].fillna("")).tolist() 
                                senti_cv.fit(texts) 
                                
                                # [수정] 'Good' 글의 평균 S/I 점수 계산 및 NaN 방어
                                df_work_senti = compute_sentiment_SI(df_work, senti_cv, lex_dict)
                                avg_s_good = df_work_senti[df_work_senti['quality_label'] == 'good']['S'].mean()
                                avg_i_good = df_work_senti[df_work_senti['quality_label'] == 'good']['I'].mean()

                                # [수정] NaN인 경우 0.0으로 폴백
                                target_s_val = float(avg_s_good) if pd.notna(avg_s_good) else 0.0
                                target_i_val = float(avg_i_good) if pd.notna(avg_i_good) else 0.0
                                
                                st.session_state['sentiment_pack'] = {
                                    'lex': lex_dict, 
                                    'cv': senti_cv,
                                    'target_s': target_s_val, 
                                    'target_i': target_i_val
                                }
                                st.session_state['lex_file_name'] = f_lex.name
                                st.success(f"감성 분석기(S/I)가 TAB1을 위해 저장되었습니다. (목표 S: {target_s_val:.2f}, 목표 I: {target_i_val:.2f})")
                        except Exception as e:
                            st.error(f"감성 사전 처리 오류: {e}")
                            st.session_state['sentiment_pack'] = None
                else:
                    st.session_state['sentiment_pack'] = None 

                # [수정] 분석 완료 플래그 및 시각화용 데이터 저장
                st.session_state['analysis_done'] = True
                st.session_state['df_work_for_viz'] = df_work.copy()
                st.session_state['df_m_raw_for_viz'] = df_m_raw.copy()
                st.session_state['topic_labels_for_viz'] = topic_labels
                st.rerun() 
                
                # ----------------- 분석 실행 블록 끝 -----------------

            # [수정] 시각화 블록 전체를 'analysis_done' 플래그 기반으로 밖으로 이동
            if st.session_state.get('analysis_done', False):
                
                # [수정] 세션에서 시각화용 데이터 로드
                df_work_viz = st.session_state.get('df_work_for_viz')
                df_m_raw_viz = st.session_state.get('df_m_raw_for_viz')
                topic_labels_viz = st.session_state.get('topic_labels_for_viz', {})
                clf_pack_viz = st.session_state.get('clf_pack') 
                senti_pack_viz = st.session_state.get('sentiment_pack')

                if df_work_viz is None or df_m_raw_viz is None or topic_labels_viz is None or clf_pack_viz is None:
                    st.error("시각화 데이터 로드 실패. 분석을 다시 실행해주세요.")
                    st.stop()

                st.markdown("---")
                st.header("🔬 추가 분석 시각화")

                # [수정] 토픽 필터
                topic_names_list = ["전체 (All)"] + [v.get('name', k) for k,v in topic_labels_viz.items()]
                filter_topic_name = st.selectbox("🔬 시각화 토픽 필터", topic_names_list)

                # [수정] 필터링된 데이터프레임 생성 (df_viz)
                topic_names_map = {int(k.split(' ')[1]): v.get('name', k) for k, v in topic_labels_viz.items()}
                
                if 'topic_name' not in df_work_viz.columns:
                     df_work_viz['topic_name'] = df_work_viz['topic'].map(topic_names_map).fillna('기타')

                if filter_topic_name == "전체 (All)":
                    df_viz = df_work_viz
                else:
                    df_viz = df_work_viz[df_work_viz['topic_name'] == filter_topic_name]

                # [시각화 1] 토픽별 성과 (이 차트는 필터 적용 안함)
                st.subheader("A. 주제별 성과 분포")
                try:
                    fig_topic_box = px.box(
                        df_work_viz, 
                        x='topic_name', 
                        y='engagement', 
                        color='topic_name',
                        title='주제(토픽)별 콘텐츠 매력 점수(Total Engagement) 분포',
                        labels={'topic_name': '주제명', 'engagement': '콘텐츠 매력 점수(총합)'}
                    )
                    st.plotly_chart(fig_topic_box, use_container_width=True)
                except Exception as e:
                    st.error(f"토픽 성과 시각화 실패: {e}")

                # [시각화 2] 성과 핵심 키워드 (전체 주제 기준)
                st.subheader("B. 성과 예측 핵심 키워드 (전체 주제 기준)")
                st.caption("└ 이 차트는 '전체 주제'에 대해 학습된 **단일 모델**의 결과이므로 토픽 필터가 적용되지 않습니다.")
                if clf_pack_viz: 
                    try:
                        good_df = pd.DataFrame(clf_pack_viz['good_terms'], columns=['term', 'score'])
                        bad_df = pd.DataFrame(clf_pack_viz['bad_terms'], columns=['term', 'score'])
                        drivers_df = pd.concat([good_df, bad_df])
                        drivers_df['type'] = np.where(drivers_df['score'] > 0, '성과 우수 (Good)', '성과 저조 (Bad)')

                        fig_drivers = px.bar(
                            drivers_df.sort_values('score'), 
                            x='score', 
                            y='term', 
                            color='type',
                            orientation='h', 
                            title='콘텐츠 등급에 영향을 미치는 핵심 키워드',
                            labels={'term': '키워드', 'score': '영향력 점수 (계수)'},
                            color_discrete_map={'성과 우수 (Good)': 'blue', '성과 저조 (Bad)': 'red'}
                        )
                        fig_drivers.update_layout(yaxis_title="키워드")
                        st.plotly_chart(fig_drivers, use_container_width=True)
                    except Exception as e:
                        st.error(f"키워드 시각화 실패: {e}")

                # [시각화 3] 감성과 성과 (필터 적용)
                st.subheader(f"C. 감성(S/I)과 성과 ({filter_topic_name})")
                if f_lex is not None:
                    if senti_pack_viz and senti_pack_viz.get('cv') and senti_pack_viz.get('lex'):
                        if 'S' not in df_viz.columns:
                            with st.spinner(f"({filter_topic_name}) 감성(S/I) 점수 계산 중..."):
                                df_viz = compute_sentiment_SI(df_viz, senti_pack_viz['cv'], senti_pack_viz['lex'])
                        
                        if 'S' in df_viz.columns and df_viz['S'].abs().sum() > 0:
                            fig_senti_scatter = px.scatter(
                                df_viz, 
                                x='S', 
                                y='engagement', 
                                color='quality_label',
                                title=f'콘텐츠 감성(S)과 성과(Total Engagement) 관계 ({filter_topic_name})',
                                labels={'S': '평균 감성 점수 (S)', 'engagement': '콘텐츠 매력 점수(총합)'},
                                hover_data=['title']
                            )
                            st.plotly_chart(fig_senti_scatter, use_container_width=True)
                        else:
                            st.warning(f"'{filter_topic_name}' 토픽에서 유효한 감성 점수(S)를 계산할 수 없었습니다.")
                    else:
                        st.caption("💖 '감성 사전 CSV'를 업로드하면 감성-성과 관계 분석이 활성화됩니다.")
                else:
                    st.caption("💖 '감성 사전 CSV'를 업로드하면 감성-성과 관계 분석이 활성화됩니다.")
                
                # [시각화 4] 시간대별 성과 추세 (전체 주제 기준)
                st.subheader("D. 기간별 성과 추세 (전체 주제 기준)")
                st.caption("└ 이 차트는 '전체 주제'의 **평균** 추세이며, 토픽 필터가 적용되지 않습니다.")
                if 'period' in df_m_raw_viz.columns: 
                    try:
                        df_trend = df_m_raw_viz.groupby('period')[['views_total', 'likes', 'comments']].mean().reset_index()
                        df_trend['parsed_period'] = pd.to_datetime(df_trend['period'], errors='coerce')
                        df_trend = df_trend.dropna(subset=['parsed_period']).sort_values('parsed_period')
                        
                        if not df_trend.empty:
                            df_trend_melted = df_trend.melt(
                                id_vars='parsed_period', 
                                value_vars=['views_total', 'likes', 'comments'],
                                var_name='Metric', 
                                value_name='Average Value'
                            )
                            
                            fig_time_trend = px.line(
                                df_trend_melted, 
                                x='parsed_period', 
                                y='Average Value', 
                                color='Metric', 
                                title='기간(Period)별 평균 성과(조회/좋아요/댓글) 추세',
                                labels={'parsed_period': '기간', 'Average Value': '평균 성과 값', 'Metric': '성과 지표'}
                            )
                            st.plotly_chart(fig_time_trend, use_container_width=True)
                        else:
                            st.warning("유효한 'period' 컬럼 데이터를 찾을 수 없어 시간대별 분석을 건너뜁니다.")
                    except Exception as e:
                        st.error(f"기간(period) 컬럼 처리 중 오류: {e}")
                else:
                    st.caption("📅 성과 측정 CSV(2)에 'period' 컬럼이 있으면 시간대별 추세 분석이 활성화됩니다.")
            
        except Exception as e:
            st.error(f"데이터 처리 오류: {e}")


# ================= TAB3 =================
with TAB3:
    st.header("🛠️ LLM 파인튜닝 (골격)")
    st.info("이 탭은 LLM 파인튜닝 프로세스에 필요한 최소한의 구조를 보여줍니다. 실제 파인튜닝 API 연동 및 데이터 전처리 코드가 필요합니다.")
    st.code("""
# [TODO] 1. 학습 데이터 준비
# df_train = ... (good/bad 콘텐츠 데이터프레임)
# df_train['prompt'] = '다음 콘텐츠를 기반으로 더 높은 성과를 위한 최적의 제목을 추천해줘: ' + df_train['content']
# df_train['completion'] = df_train['title'] # (예시)

# [TODO] 2. 데이터셋 형식 변환 (JSONL)
# training_data = [{'prompt': p, 'completion': c} for p, c in zip(df_train['prompt'], df_train['completion'])]
# jsonl_output = "\n".join([json.dumps(item, ensure_ascii=False) for item in training_data])

# [TODO] 3. OpenAI 파일 업로드 및 튜닝 API 호출
# if st.button("파인튜닝 시작"):
#     try:
#         file = client.files.create(file=io.BytesIO(jsonl_output.encode('utf-8')), purpose='fine-tune')
#         job = client.fine_tuning.jobs.create(
#             training_file=file.id, model='gpt-3.5-turbo-0125' # 또는 gpt-4o-mini
#         )
#         st.success(f"파인튜닝 작업 시작됨: {job.id}")
#     except Exception as e:
#         st.error(f"파인튜닝 실패: {e}")
    """, language="python")

# ================= [신규] TAB4 =================
with TAB4:
    st.header("🔬 모델 관리자 (Admin)")
    st.info("이 탭은 `TAB2`에서 분석이 완료된 후 활성화됩니다. 현재 적용된 모델의 상태와 성능을 점검합니다.")

    # 세션에서 데이터 로드
    df_full = st.session_state.get('df_for_analysis')
    topic_bank = st.session_state.get('topic_term_bank')
    clf_pack = st.session_state.get('clf_pack')
    lda_model = st.session_state.get('lda_model')
    lda_vect = st.session_state.get('lda_vect')
    topic_labels = st.session_state.get('topic_labels', {})

    # [수정] DataFrame __nonzero__ 오류 수정: 'is None or .empty' 사용
    if df_full is None or df_full.empty or clf_pack is None or lda_model is None or topic_bank is None:
        st.error("⚠️ 데이터가 없습니다. TAB2에서 먼저 '빠른 분석' 또는 '정밀 분석'을 실행해 주세요.")
    else:
        # --- 1. 불용어 ---
        st.subheader("1. 불용어(Stopwords) 관리")
        with st.expander("현재 적용 중인 기본 불용어 목록 보기"):
            st.text(f"총 {len(STOPWORDS_KO)}개 단어:")
            st.json(STOPWORDS_KO)
        
        # [수정] '불용어 의심' 로직 변경 (고빈도 일반어)
        with st.expander("불용어 의심 단어 보기 (고빈도 일반 단어)"):
            st.markdown("토픽/성과와 관계없이 **모든 문서에 너무 자주 등장**하는 단어(예: 10% 이상)입니다. '미국' 같은 고유명사보다 **'것이다', '있다'** 같은 일반 단어가 여기 뜬다면 불용어 추가를 고려하세요.")
            with st.spinner("모든 문서에서 고빈도 일반 단어를 추출 중입니다..."):
                suspected = get_suspected_stopwords(df_full, k=50)
                if suspected:
                    st.warning("아래 단어들은 이미 기본 불용어(STOPWORDS_KO)에 포함된 것을 제외한 고빈도 단어입니다.")
                    st.text(", ".join(suspected))
                else:
                    st.info("불용어 의심 단어를 찾지 못했습니다.")

        # --- 2. 토픽 단어 은행 ---
        st.subheader("2. 토픽별 핵심 단어 은행 (TAB1 추천 기반)")
        # [수정] 단어 은행 생성 로직 변경 안내 (Chi2)
        st.markdown("""
        [수정] 이 단어 은행은 `build_topic_term_bank` (Chi2) 함수로 생성됩니다.
        - **성과 우수 단어 (Good):** 해당 토픽의 'Good' 라벨과 통계적(**카이제곱**)으로 유의미하게 연관된 단어입니다. (추천)
        - **성과 저조 단어 (Bad):** 해당 토픽의 'Bad' 라벨과 통계적(**카이제곱**)으로 유의미하게 연관된 단어입니다. (비권장)
        - **단순 빈도 단어 (All):** 성과와 무관하게 해당 토픽에서 가장 빈도가 높은 단어입니다.
        """)
        st.caption("└ 이 단어 은행은 TAB2에서 '전체 기간' 데이터로 학습된 **고정된 모델**의 결과입니다. (최근 한 달 동적 단어는 TAB1 참조)")
        
        if topic_labels:
            topic_names_map = {v.get('name', k): int(k.split(' ')[1]) for k,v in topic_labels.items()}
            selected_name = st.selectbox("확인할 토픽 선택", list(topic_names_map.keys()))
            
            if selected_name:
                selected_id = topic_names_map[selected_name]
                
                if selected_id not in topic_bank:
                     st.error(f"토픽 {selected_id}가 단어 은행에 없습니다. (TAB2 재실행 필요)")
                else:
                    bank_data = topic_bank[selected_id]
                    # [수정] status 키를 확인하여 구체적인 오류 메시지 표시
                    if bank_data.get("status") == "ok":
                        c_g, c_b, c_a = st.columns(3)
                        c_g.dataframe({"성과 우수 단어 (Good)": [w for w,s in bank_data['good'][:20]]})
                        c_b.dataframe({"성과 저조 단어 (Bad)": [w for w,s in bank_data['bad'][:20]]})
                        c_a.dataframe({"단순 빈도 단어 (All)": [w for w,s in bank_data['all'][:20]]})
                    else:
                        st.warning(f"'{selected_name}' 토픽의 단어 은행을 표시할 수 없습니다.\n\n**사유:** {bank_data.get('message', '알 수 없는 오류')}")


        # --- 3. 모델 성능 평가 ---
        st.subheader("3. 모델 성능 평가")
        st.info("버튼을 누르면 전체 데이터를 80(학습)/20(테스트)으로 분할하여 모델 성능을 재평가합니다.")
        
        if st.button("🚀 성능 평가 실행 (80/20 분할)"):
            
            # 1. 분류(Classifier) 모델 성능 평가
            st.markdown("#### A. 성과 예측 모델 (SGDClassifier) 성능")
            st.caption("목표: 'Good' / 'Bad' 라벨을 얼마나 잘 맞추는가? (분류 모델)")
            
            with st.spinner("성과 예측 모델을 80/20 데이터로 재학습 및 평가 중..."):
                try:
                    df_trainable = df_full[df_full['quality_label'] != 'medium'].copy()
                    texts = (df_trainable["title"].fillna("") + " " + df_trainable["content"].fillna("")).tolist()
                    labels = df_trainable["quality_label"].tolist()

                    X_train_txt, X_test_txt, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)
                    
                    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=30000, min_df=2, stop_words=STOPWORDS_KO)
                    X_train_vec = tfidf.fit_transform(X_train_txt)
                    X_test_vec = tfidf.transform(X_test_txt)
                    
                    clf_test = SGDClassifier(loss="log_loss", learning_rate="optimal", alpha=1e-5, random_state=42)
                    clf_test.fit(X_train_vec, y_train)
                    
                    y_pred = clf_test.predict(X_test_vec)
                    
                    # [수정] classification_report를 딕셔너리로 받아 DataFrame으로 변환
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.text("Classification Report (Test Set):")
                    st.dataframe(report_df.round(3)) # [수정] 표(DataFrame)로 표시
                    
                    cm = confusion_matrix(y_test, y_pred, labels=['good', 'bad'])
                    st.text("Confusion Matrix (Test Set):")
                    st.dataframe(pd.DataFrame(cm, index=['True: Good', 'True: Bad'], columns=['Pred: Good', 'Pred: Bad']))

                except Exception as e:
                    st.error(f"분류 모델 평가 중 오류: {e}")
            
            st.markdown("---")
            
            # 2. 토픽(LDA) 모델 성능 평가
            st.markdown("#### B. 주제 분류 모델 (LDA) 성능")
            st.caption("목표: 문서를 얼마나 일관성 있는 주제로 잘 묶었는가? (비지도 학습)")
            st.warning("""
            **중요:** LDA는 정답이 없는 비지도 학습이므로 **Accuracy(정확도)나 F1-Score를 계산할 수 없습니다.** 대신, 모델이 얼마나 '확신을 가지고' 문서를 분류했는지, 주제가 얼마나 명확히 구분되는지를 나타내는 **Perplexity(혼잡도)**를 사용합니다.
            - **Perplexity (혼잡도):** **낮을수록 좋습니다.** 모델이 테스트 데이터를 얼마나 잘 예측하는지 나타냅니다.
            """)
            
            with st.spinner("LDA 모델의 Perplexity(혼잡도)를 계산 중..."):
                try:
                    texts = (df_full["title"].fillna("") + " " + df_full["content"].fillna("")).tolist()
                    X_full = lda_vect.transform(texts)
                    
                    perplexity = lda_model.perplexity(X_full)
                    st.metric("Perplexity (혼잡도) - 전체 데이터 기준", f"{perplexity:,.2f} (낮을수록 좋음)")
                except Exception as e:
                    st.error(f"LDA 성능 평가 중 오류: {e}")

            st.markdown("---")
            st.markdown("#### C. 회귀 모델 성능 (R² / MSE)")
            st.error("본 앱은 'Good/Bad'를 맞추는 **분류(Classification) 모델**을 사용하므로, R²(Adjusted R-squared)나 MSE 같은 **회귀(Regression) 지표**는 해당되지 않습니다. 대신 위 (A)의 **정확도(Accuracy)**와 **F1-Score**를 참조하세요.")