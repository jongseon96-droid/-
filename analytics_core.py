# analytics_core.py
# 이 파일에는 Streamlit UI와 관련된 코드를 제외한 모든 상수, 클래스, 함수만 정의됩니다.

import os, io, json, time, re as regx
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import datetime
from typing import List, Dict, Tuple

# ==== OpenAI SDK ====
from openai import OpenAI
from openai import APIError, RateLimitError

# ==== ML / NLP ====
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.preprocessing import RobustScaler
from sklearn.utils import shuffle as sk_shuffle
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold # StratifiedKFold 추가

# ==== 회귀 ====
import statsmodels.api as sm
from sklearn.pipeline import Pipeline

# ================== 1. 상수 / CONFIG ==================

# [수정] 한국어 불용어 리스트 정의 (이전 내용 그대로 유지)
STOPWORDS_KO = [
    # 조사/어미 (매우 빈번)
    "입니다", "합니다", "같습니다", "있습니다", "있는", "것입니다", "했다", "등", "이", "그", "저",
    "수", "것", "및", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "에서", "으로", "하는", "을", "를", "은", "는", "이", "가", "의", "에", "와", "과", "도", "고", "라는",

    # 일반 명사 (신호 방해)
    "블로그", "포스팅", "오늘", "이번", "다양한", "관련", "내용", "정보", "정말", "바로", "지금", "생각",
    "경우", "대해", "대부분", "때문", "가지", "통해", "위해", "대한", "통한", "따라","gt", "https", "lt", "가장", "같은", "것으로", "것은", "것이", "것이다",
    "광고", "그리고", "기사", "기사를", "뉴스", "다른", "많은", "아니라",
    "어떤", "언론", "신문과방송", "이러한", "이런", "이를", "있다", "있었다",
    "지난", "지역", "콘텐츠", "콘텐츠를", "하지만", "한다","만나보세요", "2025", "없다", "위한", "the", "com", "www", "of", "news", "and", "to", "2022" ,"uk" ,"2020", "in", "1면", "높은", "또한", "나타났다", "많이",
    "naver", "한눈에", "2020년", "늘어난", "댓글", "특히", "그림", "대비", "때문에", "없는", "것을", "때문이다", "그러나", "있다는", "무슨일이", "라고", "함께", "하고", "등을",
    "어떻게", "활용", "말했다", "ap", "niemanlab", "esg", "주목받는", "강조한", "그는", "있으며",
    "blog", "nft", "kpfjra", "에서도", "quibi", "fast", "이후", "구분", "비해", "높았다", "2021","1월", "2월", "3월", "4월", "5월", "6월",
    "7월", "8월", "9월", "10월", "11월", "12월",
    "1990", "1991", "1992", "1993", "1994", "1995", "1996", "1997", "1998", "1999",
    "2000", "2001", "2002", "2003", "2004", "2005", "2006", "2007", "2008", "2009",
    "2010", "2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019",
    "2020","보니", "있고", "라는" ,"않았다", "여러", "됐다", "우리가", "없었다", "좋은", "나는","공영방송사의", "기사는", "신문과", "방송",
    "1990년", "1991년", "1992년", "1993년", "1994년", "1995년", "1996년", "1997년", "1998년", "1999년",
    "2000년", "2001년", "2002년", "2003년", "2004년", "2005년", "2006년", "2007년", "2008년", "2009년",
    "2010년", "2011년", "2012년", "2013년", "2014년", "2015년", "2016년", "2017년", "2018년", "2019년",
    "2020년", "2021년", "2022년", "2023년", "2024년", "2025년",
    # 스크린샷에서 보인 문제 단어들
    "2024", "2023", "ai", "2024년", "2023년", "ㅋㅋ", "ㅎㅎ", "했습니다", "있었습니다", "씨의", "씨는", "위에", "기자는", "기사가", "과정을", "않았습니다", "바랍니다", "믿을", "겁니다", "않았습니다", "않고", "다시", "직접", "해당", "해당",
    "겁니다", "우리", "믿을", "갈무리", "없었습니다", "필요한", "내용을", "그런", "저는", "그래서", "내가", "다시", "그렇게", "이렇게", "일을", "말을", "있을", "보면", "되는",
    "원의", "전체", "인기를", "kr", "아닌", "따라서", "쉽게", "이는", "된다", "이에", "쉽게", "또는", "재밋게", "쉽고", "번째", "받을", "아래", "않는", "됐습니다", "인해", "매우", "관련해", "한다는", "역시", "드립니다", "밝혔다", "예를", "들면", "혹은", "들어"
]

# [수정] Baseline 모델에 필요한 Feature 리스트 정의
BASELINE_FEATURES = ["img_count", "title_length", "content_length"]

# LLM 관련 설정 (이전 내용 그대로 유지)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
USE_LLM = len(OPENAI_API_KEY) > 0
client = OpenAI(api_key=OPENAI_API_KEY) if USE_LLM else None
MODEL_CHAT = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini-2024-07-18")

# [추가] 파인튜닝 시 사용할 시스템 프롬프트 (OpenAI messages 형식에 필요)
SYSTEM_PROMPT_FT = "당신은 제시된 주제와 본문을 바탕으로 독자의 참여를 극대화하는 성과형 제목을 생성하는 전문 카피라이터입니다. 당신은 오직 제목 텍스트만 출력해야 합니다."


# LLM 상태 체크
LLM_OK = False
if USE_LLM and client:
    try:
        client.models.list()
        LLM_OK = True
    except Exception:
        LLM_OK = False

# Candidate templates (폴백) (이전 내용 그대로 유지)
NUM_RE  = regx.compile(r"\b(\d+|top\s*\d+|[0-9]+분)\b", regx.I)
TIME_BANK = ["오늘", "이번 주", "주말", "지금", "방금", "이번 달", "10월", "11월", "12월"]
HOWTO_BANK = ["방-step", "베스트 프랙티스"]
ACTION_BANK = ["정리", "비교", "분석", "설명", "추천", "점검", "실법", "가이드", "체크리스트", "튜토리얼", "Step-by험"]
CTA_BANK = ["질문", "댓글", "구독", "공유", "알림", "참여"]
LIST_BANK = ["Top 5", "Top 7", "3가지", "5분 요약", "한눈에"]
BRAND_HINT = ["한양대", "오픈AI", "카카오", "구글", "MS", "네이버"]
DEFAULT_CANDIDATES = TIME_BANK + HOWTO_BANK + ACTION_BANK + CTA_BANK + LIST_BANK + BRAND_HINT

# MODE_CFG (분석 모드 설정) (이전 내용 그대로 유지)
MODE_CFG = {
    "quick": {
        "sample_n": 5000,
        "lda_topics": 0,
        "batch_size": 500,
        "n_epochs": 2,
        "clf_epochs": 1,
        "clf_batch": 500,
        "ngram_range": (1, 2),
    },
    "full": {
        "sample_n": None,
        "lda_topics": 0,
        "batch_size": 1000,
        "n_epochs": 3,
        "clf_epochs": 3,
        "clf_batch": 1000,
        "ngram_range": (1, 3),
    },
}


# ================== 2. 유틸리티 함수 ==================

def read_csv_robust(src, **kwargs) -> pd.DataFrame:
    """[단일 파일 로드] UploadedFile/bytes/path/file-like 모두 지원. 인코딩과 구분자 자동 재시도."""
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

def categorize_term(t: str) -> str:
    """단어를 유형별로 분류"""
    t_low = t.lower()
    if NUM_RE.search(t_low) or any(x in t for x in LIST_BANK): return "숫자/리스트"
    if any(k in t for k in TIME_BANK): return "시간표현"
    if any(k in t for k in HOWTO_BANK): return "How-to/가이드"
    if any(k in t for k in CTA_BANK): return "질문/CTA"
    if any(k in t for k in ACTION_BANK): return "행동동사/행위"
    if regx.match(r"[A-Z][a-zA-Z0-9]+", t) or "대" in t or "대학" in t or any(b in t for b in BRAND_HINT):
        return "고유명사/브랜드"
    return "기타"

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
    """article_id 컬럼 정규화"""
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

def prepare_by_mode(df_in: pd.DataFrame, mode_cfg: dict, lda_topics_ui: int):
    """분석 모드에 따른 데이터 샘플링 및 설정 반환 (LDA 부분만 처리)"""
    if mode_cfg["sample_n"]:
        n = min(mode_cfg["sample_n"], len(df_in))
        df_work = df_in.sample(n=n, random_state=42).reset_index(drop=True)
    else:
        df_work = df_in.copy()

    n_topics = mode_cfg["lda_topics"] if mode_cfg["lda_topics"] > 0 else int(lda_topics_ui)

    lda_kwargs = dict(
        n_topics=n_topics,
        max_features=mode_cfg.get("max_features"),
        batch_size=mode_cfg["batch_size"],
        n_epochs=mode_cfg["n_epochs"],
    )
    clf_kwargs = dict(
        epochs=mode_cfg["clf_epochs"],
        batch_size=mode_cfg["clf_batch"],
        ngram_range=mode_cfg["ngram_range"],
    )
    return df_work, lda_kwargs, clf_kwargs


# ================== 3. 머신러닝/통계 함수 ==================

def build_engagement(df: pd.DataFrame, w_views=0.4, w_likes=0.4, w_comments=0.2) -> pd.DataFrame:
    """
    [수정] RobustScaler(중앙값/IQR) 기반 정규화된 콘텐츠 매력 점수 계산 및 Baseline 피처 엔지니어링.
    Baseline 피처(title_length, content_length, img_count)의 스케일링은 제거했습니다. (누수 방지)
    """
    df = df.copy()
    metric_cols = ["views_total", "likes", "comments"]
    
    # 1. 성과 지표 유효성 검사 및 정규화
    for c in metric_cols:
        if c not in df.columns:
            df[c] = 0
        df[c] = df[c].fillna(0)

    scaler_eng = RobustScaler(quantile_range=(25.0, 75.0))
    for c in metric_cols:
        df[c + "_rob"] = scaler_eng.fit_transform(df[[c]]).ravel()

    df["engagement"] = (
        w_views   * df["views_total_rob"] +
        w_likes   * df["likes_rob"] +
        w_comments* df["comments_rob"]
    )

    # 2. Baseline 피처 엔지니어링 (스케일링 제외)
    df["title_length"] = df["title"].fillna("").astype(str).str.len()
    df["content_length"] = df["content"].fillna("").astype(str).str.len()
    
    if "img_count" not in df.columns:
          df["img_count"] = 0 # CSV에 없으면 0으로 처리

    # BASELINE_FEATURES의 Robust Scaling은 이제 train_quality_classifier 내부에서 수행됩니다.
    
    return df

def label_quality_by_quantile(df: pd.DataFrame, col="engagement", low_q=0.33, high_q=0.66) -> pd.DataFrame:
    """분위수 기반으로 콘텐츠 품질 라벨링 (good, bad, medium)"""
    df = df.copy()
    q_low, q_high = df[col].quantile([low_q, high_q])
    def _label(x):
        if x >= q_high: return "good"
        if x <= q_low: return "bad"
        return "medium"
    df["quality_label"] = df[col].apply(_label)
    return df

def train_quality_classifier(df_train: pd.DataFrame,
                             mode: str,
                             clf_kwargs: dict,
                             lda_vect: CountVectorizer = None,
                             model_type: str = "SGDClassifier") -> Dict: 
    """
    [수정] Baseline 또는 Advanced 모드로 SGD/RandomForest 분류기를 학습. 훈련 데이터에만 스케일링을 적용합니다.
    (SGD Classifier는 partial_fit 오류 방지를 위해 class_weight='balanced'를 제거했습니다.)
    """
    if df_train.empty:
        raise ValueError("학습 데이터셋이 비어있습니다.")

    df_train = df_train[df_train["quality_label"] != "medium"].copy()
    if len(np.unique(df_train["quality_label"])) < 2:
        raise ValueError("단일 라벨만 존재하여 분류기 학습을 건너뛰웁니다.")

    y = np.array([1 if l=="good" else 0 for l in df_train["quality_label"]])
    feature_names = []
    
    scaler = RobustScaler() 

    if mode == "baseline":
        feature_cols = BASELINE_FEATURES
        X_num_scaled = scaler.fit_transform(df_train[feature_cols].values)
        X = X_num_scaled
        tfidf = None
        feature_names = feature_cols
        
    elif mode == "advanced":
        if lda_vect is None or 'topic' not in df_train.columns:
            raise ValueError("Advanced 모드는 LDA 모델 학습 및 토픽 할당이 선행되어야 합니다.")

        # 1. Tfidf (텍스트)
        texts = (df_train["content"].fillna("")).tolist()
        #texts = (df_train["title"].fillna("") + " " + df_train["content"].fillna("")).tolist()
        tfidf = TfidfVectorizer(
            ngram_range=clf_kwargs.get("ngram_range", (1, 2)),
            min_df=5, max_df=0.80, stop_words=STOPWORDS_KO
        )
        X_text = tfidf.fit_transform(texts)

        # 2. 수치 피처 (훈련 데이터에만 스케일링)
        X_num = df_train[BASELINE_FEATURES].values
        X_num_scaled = scaler.fit_transform(X_num) 
        
        # 3. 토픽 피처
        X_topic_df = pd.get_dummies(df_train['topic'], prefix='topic')
        X_topic = X_topic_df.values
        
        # 모든 피처를 통합
        X = np.hstack([X_text.toarray(), X_num_scaled, X_topic])
        
        feature_names = list(tfidf.get_feature_names_out()) + BASELINE_FEATURES + list(X_topic_df.columns)

    else:
        raise ValueError(f"알 수 없는 분류 모드: {mode}")

    # [수정] 모델 선택 및 학습 로직 (Random Forest 도입)
    if model_type == "RandomForest":
        # Random Forest: fit 사용
        clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf.fit(X, y)
    elif model_type == "SGDClassifier": # Default for Baseline Mode
        # SGDClassifier: partial_fit 사용
        clf = SGDClassifier(
            loss="log_loss", learning_rate="optimal", alpha=1e-5, random_state=42, 
            warm_start=True
        )
        # 미니배치 학습
        batch_size = clf_kwargs.get("batch_size", 2000)
        epochs = clf_kwargs.get("epochs", 3)
        classes = np.array([0,1])
        n = X.shape[0]
        idx_all = np.arange(n)
        n_batches = int(np.ceil(n / batch_size))
        
        for ep in range(epochs):
            idx_all = sk_shuffle(idx_all, random_state=42 + ep)
            for b in range(n_batches):
                bs = idx_all[b*batch_size : (b+1)*batch_size]
                Xb = X[bs]; yb = y[bs]
                clf.partial_fit(Xb, yb, classes=classes)
    else:
        raise ValueError(f"알 수 없는 모델 타입: {model_type}")

    # 키워드/피처 중요도 추출 (Tfidf 피처만)
    good_terms, bad_terms = [], []
    if mode == "advanced" and tfidf is not None and model_type == "SGDClassifier":
        tfidf_feature_count = len(tfidf.get_feature_names_out())
        coefs = clf.coef_[0][:tfidf_feature_count] 
        vocab = np.array(tfidf.get_feature_names_out())
        order = np.argsort(coefs)
        k_show = 20
        good_terms = [(vocab[i], float(coefs[i])) for i in order[::-1] if coefs[i] > 0][:k_show]
        bad_terms  = [(vocab[i], float(coefs[i]))  for i in order if coefs[i] < 0][:k_show]
    
    return {
        "clf": clf,
        "tfidf": tfidf, 
        "scaler": scaler, 
        "features": feature_names,
        "mode": mode,
        "good_terms": good_terms,
        "bad_terms": bad_terms,
        "model_type": model_type
    }

def evaluate_comparison_models(df_full: pd.DataFrame,
                               lda_vect: CountVectorizer,
                               models: List[str] = ["SGDClassifier", "LogisticRegression", "RandomForestClassifier"]):
    """
    [수정] Advanced 모드 피처셋을 사용하여 StratifiedKFold 교차 검증으로 3가지 모델의 성능을 비교합니다.
    (데이터 누수 방지를 위해 각 Fold 내부에서 수치 피처 스케일링 수행)
    """
    df_trainable = df_full[df_full['quality_label'] != 'medium'].copy().reset_index(drop=True)
    if df_trainable.empty or 'topic' not in df_trainable.columns:
        return {"error": "평가 데이터셋이 비어 있거나 토픽 정보가 없습니다."}

    # 1. 원본 데이터 준비 (스케일링 전)
    texts = (df_trainable["title"].fillna("") + " " + df_trainable["content"].fillna("")).tolist()
    y = df_trainable["quality_label"].values
    X_num_raw = df_trainable[BASELINE_FEATURES].values
    X_topic = pd.get_dummies(df_trainable['topic'], prefix='topic').values
    
    # Tfidf (텍스트) - 전체 훈련 데이터 기반으로 단어 추출 (K-Fold 밖에서 fit)
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=5, max_df=0.8, stop_words=STOPWORDS_KO)
    X_text_full = tfidf.fit_transform(texts).toarray()
    
    # 전체 피처 (Tfidf, 수치, 토픽)
    X_full = np.hstack([X_text_full, X_num_raw, X_topic])

    # 수치 피처가 시작되는 인덱스 계산
    num_start_idx = X_text_full.shape[1] 
    num_end_idx = num_start_idx + X_num_raw.shape[1]

    # 2. Stratified K-Fold 설정 (5-Fold 사용)
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    model_metrics = {m: {"Accuracy": [], "F1_Good": [], "CM_Total": np.zeros((2, 2))} for m in models}
    
    # 3. K-Fold 반복 (데이터 누수 방지 로직 실행)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_full, y)):
        
        X_train_fold, X_test_fold = X_full[train_idx].copy(), X_full[test_idx].copy()
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # ★★★ Fold 내부에서 수치 피처만 스케일링 (데이터 누수 방지) ★★★
        scaler = RobustScaler()
        scaler.fit(X_train_fold[:, num_start_idx:num_end_idx])
        
        X_train_fold[:, num_start_idx:num_end_idx] = scaler.transform(X_train_fold[:, num_start_idx:num_end_idx])
        X_test_fold[:, num_start_idx:num_end_idx] = scaler.transform(X_test_fold[:, num_start_idx:num_end_idx])
        
        # 4. 모델 학습 및 평가
        for model_name in models:
            try:
                if model_name == "SGDClassifier":
                    clf = SGDClassifier(loss="log_loss", alpha=1e-5, random_state=42 + fold_idx, class_weight='balanced')
                elif model_name == "LogisticRegression":
                    clf = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42 + fold_idx, class_weight='balanced')
                elif model_name == "RandomForestClassifier":
                    clf = RandomForestClassifier(n_estimators=100, random_state=42 + fold_idx, class_weight='balanced')
                else:
                    continue
                    
                clf.fit(X_train_fold, y_train_fold)
                y_pred = clf.predict(X_test_fold)
                
                acc = accuracy_score(y_test_fold, y_pred)
                report = classification_report(y_test_fold, y_pred, output_dict=True, zero_division=0)
                f1_good = report['good']['f1-score']
                cm_fold = confusion_matrix(y_test_fold, y_pred, labels=['good', 'bad'])
                
                model_metrics[model_name]["Accuracy"].append(acc)
                model_metrics[model_name]["F1_Good"].append(f1_good)
                model_metrics[model_name]["CM_Total"] += cm_fold
                
            except Exception as e:
                pass 
                
    # 5. 최종 결과 정리 (평균 및 전체 Confusion Matrix)
    final_results = {}
    for model_name, metrics in model_metrics.items():
        if metrics["Accuracy"]:
            final_results[model_name] = {
                "Accuracy_Mean": np.mean(metrics["Accuracy"]),
                "F1_Good_Mean": np.mean(metrics["F1_Good"]),
                "Report_DF": pd.DataFrame({
                    "Fold_Accuracy_Mean": np.mean(metrics["Accuracy"]).round(3),
                    "Fold_F1_Good_Mean": np.mean(metrics["F1_Good"]).round(3),
                    "N_Folds": N_SPLITS
                }, index=[model_name]).T,
                "CM_Total": metrics["CM_Total"],
                "Detail": f"{N_SPLITS} Fold 교차 검증 결과"
            }
        else:
            final_results[model_name] = {"error": "교차 검증 중 학습된 Fold가 없습니다."}
            
    return final_results

# build_topic_term_bank_logreg 함수는 그대로 유지
def build_topic_term_bank_logreg(df_all: pd.DataFrame,
                                 topn: int = 50,
                                 min_samples_warn: int = 50,
                                 min_samples_block: int = 10) -> dict:
    """Logistic Regression 계수를 사용하여 토픽별 단어 은행 구축"""
    bank = {}

    if 'topic' not in df_all.columns:
        return bank

    valid_topics = df_all["topic"].dropna().unique()
    unique_topics = sorted([t for t in valid_topics if pd.notna(t)])

    for t in unique_topics:
        try:
            topic_int = int(t)
        except ValueError:
            continue

        df_topic = df_all[df_all["topic"] == t]
        df_train = df_topic[df_topic["quality_label"] != "medium"]

        if len(df_train) < min_samples_block:
            bank[topic_int] = {
                "status": "error",
                "message": f"샘플 완전 부족 (N={len(df_train)}, 최소 {min_samples_block} 필요)"
            }
            continue

        warning_msg = None
        if len(df_train) < min_samples_warn:
            warning_msg = f"샘플 수(N={len(df_train)})가 권장({min_samples_warn})보다 적어 통계적 신뢰도가 낮을 수 있습니다."

        texts = (df_train["title"].fillna("") + " " + df_train["content"].fillna("")).tolist()
        y = (df_train["quality_label"] == "good").astype(int).values

        try:
            if len(np.unique(y)) < 2:
                bank[topic_int] = {"status": "error", "message": f"단일 라벨만 존재 (N={len(df_train)})"}
                continue

            tfidf = TfidfVectorizer(ngram_range=(1,1), max_features=5000, min_df=3, stop_words=STOPWORDS_KO)
            X = tfidf.fit_transform(texts)

            clf = LogisticRegression(max_iter=1000, solver="liblinear", random_state=42, class_weight='balanced')
            clf.fit(X, y)

            if not hasattr(clf, "coef_"):
                bank[topic_int] = {"status": "error", "message": "모델 학습 실패 (계수 없음)"}
                continue

            coefs = clf.coef_[0]
            vocab = np.array(tfidf.get_feature_names_out())
            order = np.argsort(coefs)

            good_terms = [(vocab[i], float(coefs[i])) for i in order[::-1] if coefs[i] > 0][:topn]
            bad_terms = [(vocab[i], float(coefs[i])) for i in order if coefs[i] < 0][:topn]

            cv_all = CountVectorizer(max_features=topn, min_df=3, stop_words=STOPWORDS_KO)
            X_all = cv_all.fit_transform(texts)
            counts = np.asarray(X_all.sum(axis=0)).ravel()
            vocab_all = np.array(cv_all.get_feature_names_out())
            order_all = np.argsort(counts)[::-1]
            all_terms = [(vocab_all[i], float(counts[i])) for i in order_all]

            bank[topic_int] = {
                "good": good_terms,
                "bad": bad_terms,
                "all": all_terms,
                "status": "ok",
                "message": f"성공 (N={len(df_train)})",
                "warning": warning_msg
            }
        except Exception as e:
            bank[topic_int] = {
                "status": "error",
                "message": f"모델 학습 실패: {e}"
            }
    return bank

# infer_topic_for_text (이전 내용 그대로 유지)
def infer_topic_for_text(txt: str,
                         vect: CountVectorizer,
                         lda_model: LatentDirichletAllocation) -> Tuple[int, np.ndarray]:
    """텍스트에 대한 토픽 추론"""
    Xd = vect.transform([txt if isinstance(txt, str) else ""])
    dist = lda_model.transform(Xd)[0]
    return int(dist.argmax()), dist

# get_topic_keywords_from_bank (이전 내용 그대로 유지)
def get_topic_keywords_from_bank(bank: dict, topic_id: int, k_each: int = 30) -> Dict[str, List[Tuple[str, float]]]:
    """주제 ID에 해당하는 'good'/'all' 키워드를 (단어, 점수) 튜플 리스트로 반환"""
    if topic_id not in bank or bank[topic_id].get("status") != "ok":
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


# llm_rerank_or_generate (수정된 최종 버전)
def llm_rerank_or_generate(
    draft_title: str,
    draft_body: str,
    candidates: List[str],
    topic_name: str,
    topk: int = 8,  # 리랭커 모드에서 사용, 파인튜닝 모드에서 무시
    audience: str = "혼합",
    tone: str = "분석적",
    temperature: float = 0.5,
    use_finetuned: bool = False,
    ft_model_id: str = MODEL_CHAT
) -> List[Dict]:
    """LLM을 이용해 제목 후보를 생성하거나 (파인튜닝), 통계 기반 단어를 리랭크 (기본)"""
    if not USE_LLM or client is None or not LLM_OK:
        raise RuntimeError("API를 사용할 수 없습니다: OPENAI_API_KEY/네트워크/권한을 확인하세요.")

    if use_finetuned and ft_model_id.startswith("ft:"):
        # ===== 1. 파인튜닝 모델 (제목 생성) 로직 (요청 사항 반영) =====
        topic_name_current = topic_name if topic_name != "미분류" else "일반"
        
        # [수정 1, 2, 4] 프롬프트 강화: 최대 개수 요청 및 Why/Effect 구조화된 JSON 반환 요청
        system_prompt = f"""당신은 제시된 주제와 본문을 바탕으로, 독자의 참여를 극대화하는 성과형 제목을 생성하는 전문 카피라이터입니다.
생성할 제목은 최대 20개 내외로 합니다.
각 제목에 대해 다음 형식을 가진 JSON 배열을 반환해야 합니다. 목록 번호나 코드 펜스는 절대 넣지 마세요.
[
    {{
        "term": "생성된 제목 텍스트",
        "why": "이 제목이 독자를 끌어당기는 구체적인 심리적/기술적 이유 (20자 내외, 창의적)",
        "expected_effect": "이 제목을 사용했을 때 예상되는 창의적이고 구체적인 성과 효과 (20자 내외, 창의적)"
    }},
    ... (최대 20개 내외)
]
"""
        user_prompt = f"""주제: {topic_name_current}
본문 초안: {draft_body}
---
위 본문을 바탕으로 다음 제약 조건에 맞는 제목을 가능한 한 많이(최대 20개) 생성해주세요.
- 독자수준: {audience}
- 톤/스타일: {tone}
- 형식: 반드시 요청된 JSON 배열 형식만 따르세요.
"""

        resp = client.chat.completions.create(
            model=ft_model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            n=1,
            temperature=temperature,
            response_format={"type": "json_object"}
        )

        raw = (resp.choices[0].message.content or "").strip()
        
        try:
            # 안전한 JSON 파싱 함수 사용
            data_obj = _parse_json_safely(raw)
            titles_list = data_obj if isinstance(data_obj, list) else (data_obj.get("items") if isinstance(data_obj, dict) else [])
        except Exception as e:
            # JSON 파싱 실패 시, 텍스트 응답을 줄 바꿈으로 분리하여 폴백 처리
            titles_list = [{"term": line.strip()} for line in raw.split('\n') if line.strip()]
        if titles_list is None:
            titles_list = []
        recs = []
        seen_terms = set()
        for i, item in enumerate(titles_list):
            title_text = item.get('term', '').strip()
            if not title_text or title_text in seen_terms:
                continue
            why_text = item.get('why', "파인튜닝된 모델이 분석한 결과, 이 제목은 높은 성과를 낼 가능성이 있습니다.") 
            effect_text = item.get('expected_effect', "독자의 호기심을 자극하여 클릭률을 획기적으로 높일 수 있습니다.")
            
            if title_text:
                recs.append({
                    "term": title_text,
                    "category": f"AI 생성 제목 {i+1}",
                    "why": why_text,
                    "where_to_add": "제목", 
                    "insertion_example": "", # [수정 3] 적용 예시 제거
                    "expected_effect": effect_text, 
                    "cautions": "원본 모델의 창의성이 반영되어 문맥을 재검토해야 할 수 있습니다."
                })
                seen_terms.add(title_text) # 중복 방지 로직 추가
        return recs # <-- IF 블록 종료

    else: # <-- ELSE 블록 시작 (리랭커 모드)
        # ===== 2. 기본 LLM (리랭커) 로직 (기존 로직 유지, 출력 형식만 수정) =====
        cand = [c.strip() for c in candidates if str(c).strip()]
        cand_unique = list(dict.fromkeys(cand))[:500]
        if not cand_unique:
            raise RuntimeError("후보 단어가 비어있습니다. (통계 기반 추천 단어 없음)")

        # [수정] 리랭커 모드용 프롬프트도 why/effect를 창의적으로 요청하도록 수정
        sys_prompt = (
            "너는 한국어 콘텐츠 편집 어시스턴트다. 반드시 JSON 객체만 출력한다. "
            "초안은 {'title': '...', 'body': '...'} JSON 객체로 제공된다. 'title'과 'body'를 명확히 구분하여 분석해야 한다. "
            "객체는 {'items': [...]} 형식이며, 각 항목은 "
            "{term, why, where_to_add, expected_effect, cautions} 키를 가진다. " # insertion_example 제거
            "where_to_add는 반드시 ['제목'] 하나여야 한다. ('소제목', '첫 120자' 등 다른 값은 절대 사용 금지) "
            "반드시 '후보 풀'에 있는 단어만 사용."
            "why와 expected_effect는 창의적이고 구체적으로 작성해야 한다." 
        )
        user_payload = {
            "goal": f"초안 문맥을 보존하며 후보 풀에서만 Top-{topk} 선별", # topk는 리랭커 모드에서 사용
            "constraints": [
                "후보 밖 단어/동의어 금지",
                "문맥 어긋나는 삽입 예시 금지",
                "중복 의미 추천 최소화",
                "where_to_add는 '제목'만 허용.",
                f"독자수준={audience}",
                f"톤={tone}"
            ],
            "candidates": cand_unique,
            "draft": {
                "title": draft_title,
                "body": draft_body[:6000]
            },
            # 리랭커 모드에서는 insertion_example을 생성하지 않도록 요청
            "return_format": [
                {"term":"...", "why":"창의적인 추천 이유", "where_to_add":"제목",
                 "expected_effect":"창의적인 예상 효과", "cautions":"..."}
            ]
        }
        
        model_name = "gpt-4o-mini-2024-07-18"
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
        data_obj = _parse_json_safely(raw)
        data = data_obj.get("items") if isinstance(data_obj, dict) else data_obj
        if not isinstance(data, list):
            raise ValueError("JSON 형식 오류: 배열(items)이 아님")

        allowed = set(cand_unique)
        recs = []
        for item in data:
            term = str(item.get("term","")).strip()
            where = str(item.get("where_to_add","")).strip()
            # [수정] where_to_add 검증을 '제목'만 허용하도록 변경
            if not term or term not in allowed or where != '제목': 
                continue
            recs.append({
                "term": term,
                "category": categorize_term(term),
                "why": str(item.get("why","")).strip(),
                "where_to_add": where,
                "insertion_example": "", # [수정 3] 적용 예시 제거 요청 반영 (리랭커 모드에서도)
                "expected_effect": str(item.get("expected_effect","")).strip(),
                "cautions": str(item.get("cautions","")).strip(),
            })
            if len(recs) >= topk:
                break
        return recs # <-- ELSE 블록 종료

# run_lda_topics_streaming (이전 내용 그대로 유지)
def run_lda_topics_streaming(
    texts: List[str],
    n_topics: int = 10,
    max_features: int = 5000, 
    batch_size: int = 1000,
    n_epochs: int = 3,
    progress_callback=None 
):
    """온라인 학습 기반 LDA 주제 분석"""
    vect = CountVectorizer(
        min_df=0.01,
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

    t0 = time.time(); step = 0
    prog = progress_callback(0.0, text="LDA 주제 분석 학습 중…") if progress_callback else None

    idx_all = np.arange(n_samples)
    for epoch in range(n_epochs):
        idx_all = sk_shuffle(idx_all, random_state=42 + epoch)
        for b in range(n_batches):
            bs = idx_all[b * batch_size : (b + 1) * batch_size]
            Xb = X[bs]
            lda.partial_fit(Xb)

            step += 1
            if prog:
                frac = step / total_steps
                elapsed = time.time() - t0
                sec_per_step = elapsed / max(step, 1)
                remain = sec_per_step * (total_steps - step)
                prog.progress(
                    frac, text=f"LDA 학습 {frac*100:.1f}% | 경과 {elapsed:,.0f}s | 남음 ~{remain:,.0f}s"
                )

    W = lda.transform(X)
    if prog: prog.empty()
    df_topic = pd.DataFrame({"topic": W.argmax(axis=1)})
    return df_topic, vect, lda, W

# train_logreg_with_progress_wrapper (이전 내용 그대로 유지)
def train_logreg_with_progress_wrapper(texts, labels, stoplist=None, ngram_range=(1,2),
                               epochs=3, batch_size=2000, k_show=20, seed=42):
    # train_logreg_with_progress 함수가 train_quality_classifier로 대체되었습니다.
    raise NotImplementedError("train_logreg_with_progress_wrapper는 train_quality_classifier로 대체되었습니다.")

# get_topic_top_words (이전 내용 그대로 유지)
def get_topic_top_words(lda, vect, topn=8):
    """LDA 결과에서 토픽별 상위 단어 추출"""
    vocab = np.array(vect.get_feature_names_out())
    topics = {}
    for k, comp in enumerate(lda.components_):
        idx = np.argsort(comp)[::-1][:topn]
        topics[f"Topic {k}"] = [str(vocab[i]) for i in idx]
    return topics

def _heuristic_topic_name(words: list[str]) -> dict:
    """휴리스틱 기반으로 토픽 이름 추론"""
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

# llm_name_topics (이전 내용 그대로 유지)
def llm_name_topics(topic_top_words: dict, model_name=MODEL_CHAT):
    """LLM을 사용하거나 휴리스틱을 사용하여 토픽 이름 라벨링"""
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

# compute_sentiment_SI (이전 내용 그대로 유지)
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

# get_sentiment_for_text (이전 내용 그대로 유지)
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

# get_recent_popular_words (이전 내용 그대로 유지)
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
        return []

# fit_ols (이전 내용 그대로 유지)
def fit_ols(y, X):
    """OLS 회귀 모델 적합"""
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

# tidy_summary (이전 내용 그대로 유지)
def tidy_summary(res: sm.regression.linear_model.RegressionResultsWrapper, max_rows=200):
    """OLS 결과를 깔끔한 DataFrame으로 변환"""
    s = []
    for name, coef, se, t, p in zip(res.params.index, res.params.values, res.bse.values, res.tvalues, res.pvalues):
        s.append({"term": name, "coef": float(coef), "se": float(se), "t": float(t), "p": float(p)})
    df = pd.DataFrame(s)
    if len(df) > max_rows:
        return df.head(max_rows)
    return df

# get_suspected_stopwords (이전 내용 그대로 유지)
def get_suspected_stopwords(df_all_data: pd.DataFrame, k: int = 50) -> List[str]:
    """토픽/성과와 무관하게 가장 자주 쓰이는 일반 단어(불용어 후보) 추출"""
    if df_all_data is None or df_all_data.empty:
        return []
    try:
        texts = (df_all_data["title"].fillna("") + " " + df_all_data["content"].fillna("")).tolist()

        cv_nostop = CountVectorizer(max_features=k,
                                    min_df=0.1,
                                    ngram_range=(1,1))
        cv_nostop.fit(texts)
        common_words = cv_nostop.get_feature_names_out()

        final_suspects = [w for w in common_words if w not in STOPWORDS_KO]
        return final_suspects
    except Exception as e:
        return []

# 시각화 함수 (이전 내용 그대로 유지)
def create_sentiment_gauge_S(s_val: float, s_target: float, lexicon_max: float = 1.0):
    """Plotly의 Indicator를 사용해 감성 점수(S) 게이지 생성"""
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = s_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
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
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=10))
    return fig

def create_sentiment_gauge_I(i_val: float, i_target: float, lexicon_max: float = 1.0):
    """Plotly의 Indicator를 사용해 감성 강도(I) 게이지 생성"""
    fig = go.Figure()
    fig.add_trace(go.Indicator(
        mode = "gauge+number",
        value = i_val,
        domain = {'x': [0, 1], 'y': [0, 1]},
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
    fig.update_layout(height=180, margin=dict(l=20, r=20, t=40, b=10))
    return fig

# @st.cache_resource를 사용하기 위한 래퍼 함수 (Streamlit progress bar를 전달)
@st.cache_resource(show_spinner=False)
def cached_lda_run_wrapper(texts_tuple, n_topics, max_features, batch_size, n_epochs):
    # run_lda_topics_streaming 내부에서 st.progress를 사용하므로 여기서는 progress_callback을 st.progress로 전달합니다.
    return run_lda_topics_streaming(
        list(texts_tuple), n_topics=n_topics,
        max_features=None,
        batch_size=batch_size, n_epochs=n_epochs,
        progress_callback=st.progress
    )

# train_logreg_with_progress_wrapper (이전 내용 그대로 유지)
def train_logreg_with_progress_wrapper(texts, labels, stoplist=None, ngram_range=(1,2),
                               epochs=3, batch_size=2000, k_show=20, seed=42):
    # train_logreg_with_progress 함수가 train_quality_classifier로 대체되었습니다.
    raise NotImplementedError("train_logreg_with_progress_wrapper는 train_quality_classifier로 대체되었습니다.")

# analytics_core.py 파일에 아래 함수를 추가합니다. (기존 evaluate_comparison_models 근처에 두는 것이 좋습니다.)

def evaluate_baseline_models(df_full: pd.DataFrame, 
                             models: List[str] = ["SGDClassifier", "LogisticRegression", "RandomForestClassifier"]):
    """
    Baseline 모드 (수치 피처만)를 사용하여 StratifiedKFold 교차 검증으로 3가지 모델의 성능을 비교합니다.
    """
    df_trainable = df_full[df_full['quality_label'] != 'medium'].copy().reset_index(drop=True)
    if df_trainable.empty:
        return {"error": "평가 데이터셋이 비어 있거나 토픽 정보가 없습니다."} # 토픽이 없어도 Baseline은 가능하지만, 동일한 오류 메시지 사용

    # 1. 원본 데이터 준비 (수치 피처만)
    y = df_trainable["quality_label"].values
    X_num_raw = df_trainable[BASELINE_FEATURES].values # BASELINE_FEATURES만 사용
    
    X_full = X_num_raw # X_full은 수치 피처만 포함

    # 2. Stratified K-Fold 설정 (5-Fold 사용)
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    model_metrics = {m: {"Accuracy": [], "F1_Good": [], "CM_Total": np.zeros((2, 2))} for m in models}
    
    # 3. K-Fold 반복 (데이터 누수 방지 로직 실행)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_full, y)):
        
        X_train_fold, X_test_fold = X_full[train_idx].copy(), X_full[test_idx].copy()
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        # ★★★ Fold 내부에서 수치 피처만 스케일링 (필수) ★★★
        # Baseline은 X_full 전체가 수치 피처이므로 전체에 스케일링 적용
        scaler = RobustScaler()
        scaler.fit(X_train_fold)
        
        X_train_fold = scaler.transform(X_train_fold)
        X_test_fold = scaler.transform(X_test_fold)
        
        # 4. 모델 학습 및 평가 (Advanced와 동일한 모델 사용)
        for model_name in models:
            try:
                if model_name == "SGDClassifier":
                    clf = SGDClassifier(loss="log_loss", alpha=1e-5, random_state=42 + fold_idx, class_weight='balanced')
                elif model_name == "LogisticRegression":
                    clf = LogisticRegression(solver='liblinear', max_iter=1000, random_state=42 + fold_idx, class_weight='balanced')
                elif model_name == "RandomForestClassifier":
                    clf = RandomForestClassifier(n_estimators=100, random_state=42 + fold_idx, class_weight='balanced')
                else:
                    continue
                    
                clf.fit(X_train_fold, y_train_fold)
                y_pred = clf.predict(X_test_fold)
                
                # 메트릭 누적
                acc = accuracy_score(y_test_fold, y_pred)
                report = classification_report(y_test_fold, y_pred, output_dict=True, zero_division=0)
                f1_good = report['good']['f1-score']
                cm_fold = confusion_matrix(y_test_fold, y_pred, labels=['good', 'bad'])
                
                model_metrics[model_name]["Accuracy"].append(acc)
                model_metrics[model_name]["F1_Good"].append(f1_good)
                model_metrics[model_name]["CM_Total"] += cm_fold
                
            except Exception as e:
                pass 
                
    # 5. 최종 결과 정리
    final_results = {}
    for model_name, metrics in model_metrics.items():
        if metrics["Accuracy"]:
            final_results[model_name] = {
                "Accuracy_Mean": np.mean(metrics["Accuracy"]),
                "F1_Good_Mean": np.mean(metrics["F1_Good"]),
                # 상세 리포트는 Advanced에서만 출력하므로 여기서는 평균만 반환
            }
        else:
            final_results[model_name] = {"error": "교차 검증 중 학습된 Fold가 없습니다."}
            
    return final_results

# ================== [신규] 자동 파인튜닝 로직 ==================

def generate_jsonl_content(df_analysis: pd.DataFrame, topic_labels: dict) -> str:
    """
    [수정 완료] df_analysis를 기반으로 GPT Fine-tuning용 'messages' 형식의 JSONL 문자열을 생성합니다.
    (OpenAI API의 최신 파인튜닝 요구 형식에 맞춤)
    """
    
    # 1. 'good' 품질 콘텐츠만 필터링
    df_good = df_analysis[df_analysis['quality_label'] == 'good'].copy()
    if df_good.empty:
        raise ValueError("Good 콘텐츠가 충분하지 않아 학습 데이터셋을 생성할 수 없습니다.")
        
    # 2. 토픽 이름 매핑
    def map_topic_name(topic_id):
        key = f"Topic {topic_id}"
        return topic_labels.get(key, {}).get('name', '일반')

    df_good['topic_name'] = df_good['topic'].apply(map_topic_name)
    
    jsonl_data = []
    
    for index, row in df_good.iterrows():
        topic_name = row['topic_name']
        content = str(row['content']) if pd.notna(row['content']) else ''
        title = str(row['title']) if pd.notna(row['title']) else ''
        
        # 3. GPT Fine-tuning을 위한 'messages' 형식으로 데이터 구조화
        messages_array = [
            {"role": "system", "content": SYSTEM_PROMPT_FT},                             # 시스템 역할
            {"role": "user", "content": f"주제: {topic_name}\n본문: {content}"},         # 사용자 입력 (프롬프트)
            {"role": "assistant", "content": title}                                     # 모델의 기대 출력 (완성)
        ]
        
        jsonl_data.append({"messages": messages_array})

    # 4. JSON Lines 문자열로 변환 (to_json 대신 수동으로 변환)
    jsonl_str = "\n".join([json.dumps(item, ensure_ascii=False) for item in jsonl_data])
    
    return jsonl_str


def run_finetuning_job(df_analysis: pd.DataFrame, topic_labels: dict, base_model: str) -> str:
    """
    GPT 파인튜닝 데이터셋을 생성, 업로드하고 학습 작업을 시작합니다.
    성공 시 job_id를 반환합니다.
    """
    if not client or not LLM_OK:
        raise RuntimeError("OpenAI API 클라이언트가 초기화되지 않았거나 인증에 문제가 있습니다.")
    
    # 1. JSONL 데이터 생성
    jsonl_content = generate_jsonl_content(df_analysis, topic_labels)
    
    # 2. 파일을 IO 객체로 변환하여 업로드
    file_io = io.BytesIO(jsonl_content.encode('utf-8'))
    file_io.name = "llm_training_data.jsonl"
    
    # API 호출: 파일 업로드
    st.info("🚀 1/2 단계: 학습 데이터셋을 OpenAI에 업로드 중...")
    
    # try-except 블록을 사용하여 API 오류를 포착합니다.
    try:
        uploaded_file = client.files.create(
            file=file_io,
            purpose="fine-tune"
        )
        file_id = uploaded_file.id
        st.success(f"✅ 학습 파일 업로드 완료 (ID: {file_id})")
    except APIError as e:
        raise APIError(f"파일 업로드 실패: {e}")
    except Exception as e:
        raise Exception(f"파일 업로드 중 예상치 못한 오류: {e}")
    
    # API 호출: 학습 작업 시작
    st.info(f"⏳ 2/2 단계: {base_model} 모델 파인튜닝 학습 작업 시작 중...")
    try:
        job = client.fine_tuning.jobs.create(
            training_file=file_id, 
            model=base_model
        )
        return job.id
    except APIError as e:
        # 🚫 이 부분을 수정하여 원래의 예외를 그대로 전파합니다.
        #    OpenAI SDK의 APIError는 인자를 자동으로 채우므로, 인자 없이 다시 raise 하는 것이 안전합니다.
        #    (또는 raise e를 사용합니다.)
        raise e  # 원래 발생한 예외를 그대로 전파
    except Exception as e:
        # 🚫 이 부분도 APIError 인자 누락 문제를 유발할 수 있으므로, 일반 Exception으로 전파합니다.
        raise Exception(f"학습 작업 생성 중 예상치 못한 오류 발생: {e}")

# analytics_core.py 에 추가 (기존 build_topic_term_bank_logreg 함수 대신 사용 가능)
from sklearn.pipeline import Pipeline # Pipeline import 추가

def build_topic_term_bank_rf_logratio(df_all: pd.DataFrame,
                                      topn: int = 50,
                                      min_samples_block: int = 10) -> dict:
    """랜덤 포레스트 피처 중요도와 로그 비율을 사용하여 토픽별 단어 은행 구축"""
    bank = {}

    if 'topic' not in df_all.columns:
        return bank

    valid_topics = df_all["topic"].dropna().unique()
    unique_topics = sorted([t for t in valid_topics if pd.notna(t)])

    for t in unique_topics:
        try:
            topic_int = int(t)
        except ValueError:
            continue

        df_topic = df_all[df_all["topic"] == t]
        df_train = df_topic[df_topic["quality_label"] != "medium"].copy() # copy() 추가

        if len(df_train) < min_samples_block:
            bank[topic_int] = {
                "status": "error",
                "message": f"샘플 완전 부족 (N={len(df_train)}, 최소 {min_samples_block} 필요)"
            }
            continue

        texts = (df_train["content"].fillna("")).tolist() # 제목을 제외하고 본문만 사용
        y = (df_train["quality_label"] == "good").astype(int).values

        try:
            if len(np.unique(y)) < 2:
                bank[topic_int] = {"status": "error", "message": "단일 라벨만 존재"}
                continue

            # 1. TF-IDF와 RandomForest 모델 학습을 위한 파이프라인
            tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1), max_features=5000, min_df=3, stop_words=STOPWORDS_KO)
            
            # 피처 중요도를 얻기 위해 파이프라인 사용
            clf_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', max_depth=10)
            
            pipeline = Pipeline([
                ('tfidf', tfidf_vectorizer),
                ('clf', clf_rf)
            ])
            pipeline.fit(texts, y)

            # 2. 피처 중요도 추출 (RandomForest)
            importances = pipeline['clf'].feature_importances_
            vocab = np.array(pipeline['tfidf'].get_feature_names_out())

            # 3. 로그 비율(Log Ratio)을 계산하여 방향성 부여 (Good vs Bad)
            X_count_good = pipeline['tfidf'].transform(df_train[df_train['quality_label'] == 'good']['content'].fillna("")).sum(axis=0)
            X_count_bad = pipeline['tfidf'].transform(df_train[df_train['quality_label'] == 'bad']['content'].fillna("")).sum(axis=0)
            
            # 각 키워드가 Good/Bad에서 나타난 횟수
            N_good = X_count_good.A1 + 1 # +1 스무딩
            N_bad = X_count_bad.A1 + 1 # +1 스무딩
            
            # 전체 문서 수
            D_good = len(df_train[df_train['quality_label'] == 'good'])
            D_bad = len(df_train[df_train['quality_label'] == 'bad'])

            # Log Ratio (확률 비율의 로그)
            # log_ratio > 0: Good에서 상대적으로 더 자주 등장
            # log_ratio < 0: Bad에서 상대적으로 더 자주 등장
            log_ratio = np.log((N_good / D_good) / (N_bad / D_bad))

            # 4. 중요도와 로그 비율을 결합하여 순위 결정
            # 중요도가 높고 (RF) 로그 비율이 양수인 (Good 선호) 단어 순위
            combined_score_good = importances * (log_ratio > 0)
            order_good = np.argsort(combined_score_good)[::-1]
            
            # 중요도가 높고 (RF) 로그 비율이 음수인 (Bad 선호) 단어 순위
            combined_score_bad = importances * (log_ratio < 0)
            order_bad = np.argsort(combined_score_bad)[::-1]

            # 5. 최종 목록 생성 (Log Ratio 값을 Score로 사용)
            good_terms = []
            for i in order_good:
                if combined_score_good[i] > 0: # 긍정 방향성을 가진 단어만
                    good_terms.append((vocab[i], float(log_ratio[i])))
                if len(good_terms) >= topn: break

            bad_terms = []
            for i in order_bad:
                if combined_score_bad[i] > 0: # 부정 방향성을 가진 단어만
                    bad_terms.append((vocab[i], float(log_ratio[i])))
                if len(bad_terms) >= topn: break
            
            # 전체 빈도 추출 (기존 로직 유지)
            cv_all = CountVectorizer(max_features=topn, min_df=3, stop_words=STOPWORDS_KO)
            X_all = cv_all.fit_transform(df_train["content"].fillna(""))
            counts = np.asarray(X_all.sum(axis=0)).ravel()
            vocab_all = np.array(cv_all.get_feature_names_out())
            order_all = np.argsort(counts)[::-1]
            all_terms = [(vocab_all[i], float(counts[i])) for i in order_all]

            bank[topic_int] = {
                "good": good_terms,
                "bad": bad_terms,
                "all": all_terms,
                "status": "ok",
                "message": f"성공 (N={len(df_train)})",
                "warning": f"랜덤 포레스트 중요도 기반 추출. Score는 Log Ratio 값입니다."
            }
        except Exception as e:
            bank[topic_int] = {
                "status": "error",
                "message": f"모델 학습 실패: {e}"
            }
    return bank