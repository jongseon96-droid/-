# app_main.py (최종 자동화 버전)

# -*- coding: utf-8 -*-

# app_main.py
# 이 파일은 Streamlit UI와 메인 로직을 담당합니다.

import os, json, time, re as regx
import streamlit as st
import pandas as pd
import numpy as np
import datetime
from typing import List, Dict, Tuple

# analytics_core 모듈에서 모든 필요한 함수와 상수를 가져옵니다.
from analytics_core import (
    read_csv_robust, categorize_term, _parse_json_safely,
    coerce_article_id, build_topic_term_bank_logreg, # LogReg 함수는 그대로 유지
    infer_topic_for_text, get_topic_keywords_from_bank,
    llm_rerank_or_generate, build_engagement, label_quality_by_quantile,
    get_topic_top_words, llm_name_topics, compute_sentiment_SI,
    get_sentiment_for_text, get_recent_popular_words,
    get_suspected_stopwords, fit_ols, tidy_summary,
    create_sentiment_gauge_S, create_sentiment_gauge_I,
    prepare_by_mode,
    # 새로운 학습 및 비교 함수 import
    train_quality_classifier, evaluate_comparison_models,
    evaluate_baseline_models, 
    # [추가] 파인튜닝 자동화를 위한 함수 import
    run_finetuning_job,
    build_topic_term_bank_rf_logratio, # ★★★ RandomForest 기반 함수 import (TAB2에서 사용) ★★★
    # =================================
    MODE_CFG, DEFAULT_CANDIDATES, LLM_OK, client,
    cached_lda_run_wrapper, STOPWORDS_KO, MODEL_CHAT, APIError, RateLimitError,
    train_test_split, TfidfVectorizer, SGDClassifier, confusion_matrix, classification_report,
    RobustScaler,   # Scaler 객체를 사용할 수 있도록 import
    BASELINE_FEATURES # BASELINE_FEATURES 리스트 import
)
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

# ================== CONFIG (메인 앱에서만 사용하는) ==================
# [신규] 파인튜닝 설정 (완료 가정)
USE_FINETUNED_MODEL = True
FINETUNED_MODEL_ID_DEFAULT = "ft:gpt-4o-mini-2024-07-18:::CWPoHwfK" 

def require_llm():
    if not LLM_OK:
        st.error("API를 사용할 수 없습니다: OPENAI_API_KEY/네트워크/권한을 확인하세요.")
        st.stop()

# ========= Streamlit UI / Main Logic =========
def main():
    st.set_page_config(page_title="문맥형 추천 + 성과 분석 + 감성/회귀", page_icon="📝", layout="wide")
    st.title("Team 5_통계적데이터과학")

    with st.sidebar:
        st.subheader("공통 설정")
        audience = st.selectbox("주요 독자 수준", ["입문자", "전문가", "혼합"], index=2)
        tone = st.selectbox("콘텐츠 톤/스타일", ["친근", "공식", "분석적"], index=2)
        if LLM_OK: st.success("LLM 상태: ✅ 연결 OK")
        elif client: st.error("LLM 상태: ❌ 인증/권한/네트워크 오류")
        else: st.error("LLM 상태: ❌ OPENAI_API_KEY 미설정")

    # [수정] TAB4_FT (파인튜닝 관리자) 추가
    TAB1, TAB2, TAB3_ADMIN, TAB4_FT = st.tabs([
        "💡 문맥형 용어 추천",
        "📈 성과/주제/감성 분석",
        "🔬 모델 관리자 (Admin)",
        "🤖 파인튜닝 관리자 (FT)"
    ])

    # 세션 상태 초기화
    for _k, _v in [
        ("last_recs", None),
        ("last_recs_time", None),
        ("last_draft", ""),
        ("last_candidates", []),
        ("sentiment_pack", None),
        ("df_for_analysis", None),
        ("analysis_done", False),
        ("ft_model_id", FINETUNED_MODEL_ID_DEFAULT), 
        ("lda_vect", None),
        ("lda_model", None),
        ("clf_pack_base", None), 
        ("clf_pack_adv", None),   
        ("topic_term_bank", None),
        ("ft_job_id", None),
        ("comparison_results", None), # TAB3 비교 결과 저장
        ("ft_training_file", None), # [신규] 파인튜닝 학습 파일
    ]:
        st.session_state.setdefault(_k, _v)

    # 파인튜닝 모드 확인 (메인 UI 로직에서만 사용)
    DUMMY_ID = "ft:gpt-4o-mini-DUMMY_ID_INIT"
    FINETUNED_MODEL_ID_CURRENT = st.session_state.get('ft_model_id', DUMMY_ID)
    is_ft_model_ready = USE_FINETUNED_MODEL and FINETUNED_MODEL_ID_CURRENT.startswith("ft:") and FINETUNED_MODEL_ID_CURRENT != DUMMY_ID


    # ================= TAB1 =================
    with TAB1:
        st.header("1) 초안 텍스트 입력 및 분석 모드 선택")
        
        # [수정 1] Baseline / Advanced 모드 선택
        mode = st.radio(
            "분석/추천 모드 선택",
            ["Advanced Mode (토픽 + 피처)", "Baseline Mode (수치 피처만)"],
            index=0,
            horizontal=True,
            help="Advanced: 텍스트 의미(토픽)와 수치적 특징 모두 사용. Baseline: img_count, length 등 수치적 특징만 사용."
        )
        selected_clf_pack = st.session_state.get("clf_pack_adv" if mode.startswith("Advanced") else "clf_pack_base")

        # --- 제목 입력 및 길이 표시 ---
        draft_title = st.text_input("제목 (선택)", placeholder="예: 이번 주 AI 트렌드 Top 5")
        current_title_len = len(draft_title.strip())
        st.caption(f"**제목 길이:** {current_title_len}자")
        
        # --- 본문 입력 및 길이 표시 ---
        draft_body = st.text_area("본문 (초안)", height=220,
                                     placeholder="예) 1. 오픈AI의 새 모델이...")
        current_content_len = len(draft_body.strip())
        st.caption(f"**본문 길이:** {current_content_len}자")
        
        # [수정 2] 수치 피처 입력 받기 (길이는 이미 계산됨)
        st.markdown("---")
        st.subheader("1-1) 이미지 수 입력 (성과 확률 예측에 사용)")
        
        # 기본값 설정 (평균값)
        df_analysis = st.session_state.get('df_for_analysis')
        if df_analysis is not None and not df_analysis.empty:
            img_default = int(df_analysis['img_count'].mean())
        else:
            img_default = 3

        # 이미지 수만 입력받는 UI
        img_count = st.number_input("이미지 수 (img_count)", min_value=0, value=img_default, key='ui_img_count')
        st.markdown("---")


        full_draft = draft_title.strip() + " " + draft_body.strip()

        c_date, c_check = st.columns([1, 1])
        with c_date:
            ref_date = st.date_input("기준 날짜", datetime.date.today())
        with c_check:
            st.write("")
            st.write("")
            all_dates = st.checkbox("모든 날짜 선택하기 (전체 기간 분석)", value=True)

        candidates = list(DEFAULT_CANDIDATES)
        topic_id_for_draft, topic_dist = None, None
        topic_name = "미분류"

        # 세션에서 필요한 모델/데이터 로드
        topic_bank = st.session_state.get("topic_term_bank")
        lda_vect    = st.session_state.get("lda_vect")
        lda_model   = st.session_state.get("lda_model")
        senti_pack = st.session_state.get('sentiment_pack')
        df_all_data = st.session_state.get('df_for_analysis')

        # [신규] 데이터 미로드 시 경고
        if not st.session_state['analysis_done']:
            st.warning("⚠️ **데이터 미로드:** TAB2에서 CSV 업로드 및 분석을 실행하면 과거 데이터 기반의 확률, 주제, 키워드 추천이 활성화됩니다.")


        # --- 1. 토픽 추론 및 태그 표시 (Advanced Mode에서만) ---
        if full_draft.strip() and mode.startswith("Advanced") and topic_bank and lda_vect is not None and lda_model is not None:
            topic_id_for_draft, topic_dist = infer_topic_for_text(full_draft, lda_vect, lda_model)

            topic_name = f"토픽 {topic_id_for_draft}"
            topic_desc = "분석된 주제"
            lbls = st.session_state.get("topic_labels", {})
            if f"Topic {topic_id_for_draft}" in lbls:
                meta = lbls[f"Topic {topic_id_for_draft}"]
                topic_name = meta.get('name', topic_name)
                topic_desc = meta.get('desc', topic_desc)
                
            # 해당 토픽의 성과 우수 단어를 다시 로드하여 후보 단어로 사용
            topic_keywords_data = get_topic_keywords_from_bank(topic_bank, int(topic_id_for_draft), k_each=30)
            candidates = list(dict.fromkeys([w for w, s in topic_keywords_data.get("good", [])] + DEFAULT_CANDIDATES))


            st.markdown(f"**초안의 예상 주제:** <span style='background-color: #0072F0; color: white; padding: 3px 8px; border-radius: 15px; font-size: 0.9em; margin-left: 10px;'>{topic_name}</span>", unsafe_allow_html=True)
            
            # [수정 반영] 후보 단어 리스트를 Expander로 감싸서 표시
            with st.expander(f"후보 단어 ({len(candidates)}개) 펼쳐보기"):
                st.caption(f"└ {topic_desc} (토픽: {topic_id_for_draft}) 기반으로 추출된 **성과 우수 단어** 및 기본 단어 풀입니다.")
                st.code(", ".join(candidates)) # 후보 단어 목록을 code 블록으로 표시


        elif mode.startswith("Advanced"):
            st.caption("ℹ️ Advanced Mode는 TAB2에서 LDA 분석이 완료된 후 활성화됩니다.")


        # --- 2. 등급/확률 및 키워드 추천 (분류기 로드 시) ---
        if full_draft.strip() and selected_clf_pack is not None:
            clf = selected_clf_pack["clf"]
            
            # [핵심 수정 3] 예측 피처 구성 (scaler 사용)
            try:
                # 1. 수치 피처 (사용자 입력/자동 계산된 길이)
                X_num_raw = np.array([[img_count, current_title_len, current_content_len]])
                
                # 2. Scaler를 사용하여 수치 피처 변환
                scaler = selected_clf_pack.get("scaler")
                if scaler is None:
                    raise ValueError("학습된 Scaler 객체가 없습니다. TAB2에서 분석을 다시 실행해주세요.")
                    
                X_num_scaled = scaler.transform(X_num_raw)
                
                if mode.startswith("Advanced"):
                    # Advanced: Tfidf + Scaled Numerical + Topic One-Hot
                    tfidf = selected_clf_pack["tfidf"]
                    
                    # 3. 텍스트 Tfidf 피처
                    X_text = tfidf.transform([full_draft]).toarray()
                    
                    # 4. 토픽 피처 (One-Hot)
                    topic_cols = [f for f in selected_clf_pack["features"] if f.startswith('topic_')]
                    X_topic = np.zeros((1, len(topic_cols)))
                    
                    if topic_id_for_draft is not None:
                        topic_one_hot_key = f'topic_{topic_id_for_draft}'
                        if topic_one_hot_key in selected_clf_pack["features"]:
                            # Xd 구성에 맞게 토픽 원핫 인덱스 계산 (Tfidf + Scaled Num 이후)
                            tfidf_feature_count = len(tfidf.get_feature_names_out())
                            num_feature_count = len(BASELINE_FEATURES)
                            topic_one_hot_index = selected_clf_pack["features"].index(topic_one_hot_key) - (tfidf_feature_count + num_feature_count)
                            
                            if 0 <= topic_one_hot_index < len(topic_cols):
                                X_topic[0, topic_one_hot_index] = 1
                    
                    # 최종 피처 벡터: Tfidf + Scaled Numerical + Topic One-Hot
                    Xd = np.hstack([X_text, X_num_scaled, X_topic])

                else: # Baseline Mode
                    # Baseline: Scaled Numerical Feature만 사용
                    Xd = X_num_scaled

                # 예측
                if Xd.shape[1] == len(selected_clf_pack['features']):
                    proba_good = float(clf.predict_proba(Xd)[0,1])
                    label = "상 (Good)" if proba_good >= 0.5 else "하 (Bad)"
                else:
                    proba_good = 0.5
                    label = f"오류: 피처 수 불일치 ({Xd.shape[1]} vs {len(selected_clf_pack['features'])})"
                    st.warning(f"예측 피처 개수가 모델({selected_clf_pack['mode']}) 학습 피처 개수와 다릅니다. (예측: {Xd.shape[1]}, 학습: {len(selected_clf_pack['features'])})")

            except Exception as e:
                proba_good = 0.5
                label = f"예측 오류"
                st.error(f"예측 중 심각한 오류 발생: {e}")


            c1, c2 = st.columns(2)
            c1.metric(f"예상 콘텐츠 매력 등급 ({mode})", label)
            c2.metric("📈 과거 데이터 기반 성과 확률", f"{proba_good*100:.1f}%")

            st.caption(f"└ 과거 데이터(TAB2)로 학습한 **{mode} 모델**의 예측치입니다. (모델 타입: {selected_clf_pack.get('model_type', '불명')})")
            
            # --- 키워드 추천 섹션 (Advanced Mode에만 해당) ---
            if not is_ft_model_ready and mode.startswith("Advanced"):
                if all_dates:
                    if topic_id_for_draft is not None and topic_bank:
                        topic_keywords_data = get_topic_keywords_from_bank(topic_bank, int(topic_id_for_draft), k_each=10)
                        good_topic_terms = [w for w,s in topic_keywords_data.get("good", [])]
                        if good_topic_terms:
                            with st.expander(f"✅ **'{topic_name}' 주제**의 **전체 기간** 성과 우수 단어 (추천)"):
                                st.markdown(f"**이유:** 과거 이 주제(`{topic_name}`)의 콘텐츠 중 **높은 성과**를 낸 문서에서 자주 발견된 단어들입니다.")
                                st.info(", ".join(good_topic_terms))
                            # candidates 리스트가 이미 위에서 업데이트되었으므로 여기서는 pass
            
            st.divider()

            # --- 3. 감성 점수 (감성 사전 로드 시) ---
            if senti_pack and senti_pack.get('cv') and senti_pack.get('lex'):
                senti_s, senti_i = get_sentiment_for_text(full_draft, senti_pack)
                target_s = senti_pack.get('target_s')
                target_i = senti_pack.get('target_i')

                col1, col2 = st.columns(2)
                with col1:
                    st.plotly_chart(create_sentiment_gauge_S(senti_s, target_s), use_container_width=True)
                with col2:
                    st.plotly_chart(create_sentiment_gauge_I(senti_i, target_i), use_container_width=True)

                if target_s is not None and target_i is not None:
                    st.markdown(f"**🎯 목표 점수** (Good 콘텐츠 평균): **S (점수): {target_s:.2f}** | **I (강도): {target_i:.2f}**")
            
            st.divider()

        # --- 4. LLM 추천/생성 (메인 로직 스위치) ---
        if is_ft_model_ready:
            st.subheader("2) 🤖 AI 제목 생성기 (파인튜닝 모델 사용 중)")
            st.caption(f"파인튜닝된 모델({FINETUNED_MODEL_ID_CURRENT[:20]}...)이 분석 대신 **제목을 직접 생성**합니다. (최대 20개 내외)")
            
            # [수정 반영] 파인튜닝 모드에서 topk 슬라이더 제거 및 고정 값 설정
            topk = 20
            
            btn_label = "✨ AI 제목 생성 시작"
        else:
            st.subheader("2) LLM 리랭커 (통계 기반 후보 사용 중)")
            topk = st.slider("추천 개수 (Top-K)", 3, 15, 8)
            btn_label = "✨ 문맥형 용어 추천 생성"

        btn = st.button(btn_label, disabled=not LLM_OK)

        if btn:
            require_llm()
            if not full_draft.strip():
                st.warning("제목이나 본문 초안 텍스트를 입력하세요.")
            elif mode.startswith("Baseline"):
                st.warning("Baseline Mode에서는 키워드 추천 로직을 실행하지 않습니다. Advanced Mode로 전환하거나, 파인튜닝 모델을 사용하세요.")
            else:
                with st.spinner("LLM이 제목을 생성/선별 중입니다..."):
                    try:
                        recs = llm_rerank_or_generate(
                            draft_title=draft_title,
                            draft_body=draft_body,
                            candidates=candidates,
                            topic_name=topic_name,
                            topk=topk,
                            audience=audience,
                            tone=tone,
                            temperature=0.2,
                            use_finetuned=is_ft_model_ready,
                            ft_model_id=FINETUNED_MODEL_ID_CURRENT
                        )
                        st.session_state["last_recs"] = recs
                        st.session_state["last_recs_time"] = time.strftime("%Y-%m-%d %H:%M:%S")

                        st.success("추천 완료!")
                        st.session_state["last_draft"] = full_draft
                        st.session_state["last_candidates"] = list(candidates)
                    except (APIError, RateLimitError) as e:
                        st.error(f"OpenAI API 오류 (할당량, 인증 등): {e}")
                    except Exception as e:
                        st.error(str(e))


        if st.session_state.get("last_recs"):
            if is_ft_model_ready:
                st.subheader("✅ AI 생성 제목 후보 (Top-K)")
                result_label = "AI 생성 제목 후보"
            else:
                st.subheader("✅ LLM 리랭커 추천 단어")
                result_label = "문맥 추천 단어 후보"

            st.markdown(f"**총 {len(st.session_state['last_recs'])}개의 {result_label}가 있습니다.**")
            st.markdown("---")

            # [수정된 부분: Expander를 유지하고 내용을 변경]
            for i, r in enumerate(st.session_state["last_recs"], 1):
                term_text = r.get('term', '(용어)').strip()
                category_text = r.get('category', '')
                why_text = r.get('why', '설명 없음')
                example_text = r.get('insertion_example', '예시 없음')
                expected_effect = r.get('expected_effect', '정보 없음')

                st.markdown(f"**{i}. {term_text}** (추천 위치: {r.get('where_to_add', '위치 불명')})")
                
                # '자세히 보기' Expander 유지
                with st.expander(f"자세히 보기: {term_text}"):
                    
                    # 파인튜닝 모드가 아닐 때만 분류 표시
                    if not is_ft_model_ready:
                        st.markdown(f"**분류:** `{category_text}`")
                    
                    # [핵심] 추천 맥락/이유 강조 (Why)
                    st.markdown(f"**💡 추천 맥락/이유 (Why):**")
                    st.info(f"**{why_text}**") 
                    
                    # 적용 예시 필드는 analytics_core에서 제거되었으므로 출력 로직을 건너뜁니다.
                    
                    # 예상 효과 강조
                    st.markdown(f"**📈 예상 효과:** **{expected_effect}**")
                
                st.markdown("---")

            st.caption(f"• 최종 추천은 {st.session_state.get('last_recs_time', 'N/A')}에 생성되었습니다.")

        else:
            st.info("아직 생성된 추천이 없습니다. 위 버튼으로 먼저 생성하세요.")

        st.markdown("---")
        st.caption("• 추천 용어는 TAB2에서 분석한 데이터 기반으로 생성된 '토픽별 핵심 단어' 풀에서 선별됩니다.")

    # ================= TAB2 =================
    with TAB2:
        st.header("📊 데이터 업로드 및 성과 분석")
        
        # [수정] 단일 파일 업로더로 변경
        f_data = st.file_uploader(
            "📝 (1) 콘텐츠 및 성과 데이터 CSV (article_id, title, content, date, views_total, likes, comments, img_count, title_length, content_length 포함)", 
            type=["csv"], 
            key="data"
        )
        
        st.markdown("---")
        st.subheader("⚙️ 분석 설정")
        c3, c4, c5 = st.columns(3)
        lda_topics = c3.number_input("주제 분류 개수 (LDA 토픽 수)", min_value=5, max_value=40, value=10, step=1)
        c4.markdown("**콘텐츠 매력 점수 가중치** (총합 1.0)")
        wv = c4.slider("조회수 가중치", 0.0, 1.0, 0.4, 0.05, key="wv_slider")
        wl = c4.slider("좋아요 가중치", 0.0, 1.0, 0.4, 0.05, key="wl_slider")
        wc = c4.slider("댓글 가중치", 0.0, 1.0, 0.2, 0.05, key="wc_slider")

        f_lex = c5.file_uploader("💖 감성 사전 CSV (선택: word,score)", type=["csv"], key="lex")

        # 파일 변경 감지 및 데이터 로드 (분석 버튼 밖에 위치)
        if f_data is not None:
            try:
                is_new_file = False
                if st.session_state.get('f_data_name') != f_data.name:
                    st.session_state['analysis_done'] = False
                    st.session_state['f_data_name'] = f_data.name
                    is_new_file = True
                
                # [수정] 단일 파일 로드 로직
                if not st.session_state['analysis_done'] or is_new_file:
                    with st.spinner("새로운 파일 감지. 데이터를 로드하고 전처리합니다..."):
                        df_raw = coerce_article_id(read_csv_robust(f_data))
                        
                        required_cols = ["title", "content", "views_total", "likes", "comments", "img_count"]
                        missing = [c for c in required_cols if c not in df_raw.columns]
                        if missing:
                            st.error(f"필수 컬럼 누락: {', '.join(missing)} 이(가) CSV 파일에 포함되어야 합니다.")
                            st.stop()

                        df_raw["article_id"] = df_raw["article_id"].astype(str)
                        for col in ["views_total", "likes", "comments", "img_count"]:
                            df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce').fillna(0)

                        if 'date' in df_raw.columns:
                            df_raw['date'] = pd.to_datetime(df_raw['date'], errors='coerce')
                        
                        df = build_engagement(df_raw, w_views=wv, w_likes=wl, w_comments=wc)
                        df = label_quality_by_quantile(df, col="engagement", low_q=0.33, high_q=0.66)

                        st.session_state['df_for_analysis'] = df.copy()
                        st.session_state['df_m_raw_for_viz'] = df_raw.copy()
                        st.success(f"데이터 로드 및 전처리 완료: {len(df)} 건. (분석 버튼을 눌러주세요)")
                
                # 등급 확인 표시
                df_full_display = st.session_state.get('df_for_analysis')
                if df_full_display is not None and not df_full_display.empty:
                    st.subheader("1. 콘텐츠 등급 확인")
                    st.caption("콘텐츠 매력 점수(Total Engagement)를 기준으로 상위 33%는 '상 (good)', 하위 33%는 '하 (bad)'로 분류했습니다.")
                    grade_counts = df_full_display["quality_label"].value_counts().rename({"good": "상 (Good)", "medium": "중 (Medium)", "bad": "하 (Bad)"})
                    st.dataframe(grade_counts.to_frame(name="콘텐츠 수"), use_container_width=True)
                else:
                    st.info("CSV 파일을 업로드하면 데이터 등급을 확인할 수 있습니다.")


                colm1, colm2 = st.columns(2)
                do_quick = colm1.button("⚡️ 빠른 분석 (샘플/경량 모델)", use_container_width=True)
                do_full = colm2.button("🔬 정밀 분석 (전체/고정밀 모델)", use_container_width=True)

                if do_quick or do_full:
                    # ----------------- 분석 실행 블록 시작 -----------------
                    mode = "quick" if do_quick else "full"
                    cfg = MODE_CFG[mode]

                    df_full_for_prep = st.session_state.get('df_for_analysis')
                    if df_full_for_prep is None or df_full_for_prep.empty:
                        st.error("데이터 로드 오류: df_for_analysis가 비어있습니다. 파일을 다시 업로드하세요.")
                        st.stop()

                    df_work, lda_kw, clf_kw = prepare_by_mode(df_full_for_prep, cfg, lda_topics)
                    
                    # ===== LDA (Advanced Mode용) =====
                    st.subheader("2. 주제(토픽) 분류 및 분석 (Advanced Mode용)")
                    with st.spinner(f"LDA({mode}) 주제 분석 실행 중…"):
                        df_sig = tuple(df_work["content"].fillna("").tolist())
                        df_topic, vect, lda, W = cached_lda_run_wrapper(df_sig, **lda_kw)
                    
                    df_work["topic"] = df_topic["topic"]
                    st.write("주제 분류 결과 (샘플):", df_work[["article_id","topic","title"]].head(10))

                    topics_top_words = get_topic_top_words(lda, vect, topn=8)
                    with st.spinner("LLM/휴리스틱으로 토픽 라벨링 중..."):
                        topic_labels = llm_name_topics(topics_top_words)

                    st.session_state["topic_labels"] = topic_labels
                    st.session_state["lda_vect"] = vect
                    st.session_state["lda_model"] = lda

                    # 전체 데이터에 토픽 분류 적용 (TAB1/TAB3에서 사용)
                    if 'df_for_analysis' in st.session_state and st.session_state['df_for_analysis'] is not None:
                        # [수정] 제목 제외, 본문만 텍스트로 사용
                        full_texts = (st.session_state['df_for_analysis']["content"].fillna("")).tolist()
                    #full_texts = (st.session_state['df_for_analysis']["title"].fillna("") + " " + st.session_state['df_for_analysis']["content"].fillna("")).tolist()
                    full_X = vect.transform(full_texts)
                    full_topics = lda.transform(full_X).argmax(axis=1)
                    st.session_state['df_for_analysis']['topic'] = full_topics
                    st.info("전체 데이터에 토픽 분류 적용 완료.")

                    # ===== 분류기 학습 (Baseline / Advanced) =====
                    st.subheader("3. 콘텐츠 등급 예측 모델 학습")
                    df_trainable = st.session_state['df_for_analysis'] 
                    
                    # 3-1. Baseline 모델 학습 (SGD 사용)
                    with st.spinner(f"3-1. Baseline 모델 (수치 피처, SGD) 학습 중..."):
                        clf_pack_base = train_quality_classifier(df_trainable, "baseline", clf_kw, model_type="SGDClassifier")
                        st.session_state["clf_pack_base"] = clf_pack_base
                        st.success("Baseline 모델 학습 완료!")
                        
                    # 3-2. Advanced 모델 학습 (RandomForest 사용)
                    with st.spinner(f"3-2. Advanced 모델 (토픽+피처, RandomForest) 학습 중..."):
                        # ★★★ Advanced Mode는 RandomForest 사용 ★★★
                        clf_pack_adv = train_quality_classifier(df_trainable, "advanced", clf_kw, vect, model_type="RandomForest")
                        st.session_state["clf_pack_adv"] = clf_pack_adv
                        st.success("Advanced 모델 (RandomForest) 학습 완료!")
                        
                    # ===== 토픽 단어 은행 구축 (Advanced Mode용) =====
                    st.subheader("4. 토픽별 핵심 단어 은행 구축 (TAB1 추천 기반)")
                    with st.spinner("토픽별 성과 우수/저조 단어 분석 중…"):
                        # ★★★ [수정] RandomForest 기반 함수 호출로 대체 ★★★
                        topic_term_bank = build_topic_term_bank_rf_logratio(st.session_state['df_for_analysis'], topn=50) 
                    
                        st.session_state["topic_term_bank"] = topic_term_bank
                    st.success("토픽 기반 용어 은행(RandomForest/LogRatio) 구축 완료! (TAB1 Advanced Mode에서 활용 가능)")

                    # ===== 감성 분석기 생성 로직 (TAB1용) =====
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
                                    senti_cv = TfidfVectorizer(min_df=1, stop_words=STOPWORDS_KO)
                                    texts = (df_full_for_prep["title"].fillna("") + " " + df_full_for_prep["content"].fillna("")).tolist()
                                    senti_cv.fit(texts)

                                    df_work_senti = compute_sentiment_SI(df_work, senti_cv, lex_dict)
                                    avg_s_good = df_work_senti[df_work_senti['quality_label'] == 'good']['S'].mean()
                                    avg_i_good = df_work_senti[df_work_senti['quality_label'] == 'good']['I'].mean()

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


                    # 분석 완료 플래그 및 시각화용 데이터 저장
                    st.session_state['analysis_done'] = True
                    st.session_state['df_work_for_viz'] = df_work.copy()
                    st.session_state['topic_labels_for_viz'] = topic_labels
                    st.rerun()

                # --- 시각화 블록 (분석 완료 시에만 실행) ---
                if st.session_state.get('analysis_done', False):
                    df_work_viz = st.session_state.get('df_work_for_viz')
                    topic_labels_viz = st.session_state.get('topic_labels_for_viz', {})
                    clf_pack_adv_viz = st.session_state.get('clf_pack_adv')
                    senti_pack_viz = st.session_state.get('sentiment_pack')

                    if df_work_viz is None or topic_labels_viz is None or clf_pack_adv_viz is None:
                        st.error("시각화 데이터 로드 실패. 분석을 다시 실행해주세요.")
                        st.stop()

                    st.markdown("---")
                    st.header("🔬 추가 분석 시각화")

                    topic_names_map = {int(k.split(' ')[1]): v.get('name', k) for k, v in topic_labels_viz.items()}
                    if 'topic_name' not in df_work_viz.columns:
                        df_work_viz['topic_name'] = df_work_viz['topic'].map(topic_names_map).fillna('기타')
                    
                    # A, B 섹션에 적용될 토픽 필터
                    topic_names_list = ["전체 (All)"] + sorted(df_work_viz['topic_name'].unique().tolist())
                    filter_topic_name = st.selectbox("🔬 시각화 토픽 필터 (A, B 섹션에 적용)", topic_names_list)

                    if filter_topic_name == "전체 (All)":
                        df_viz = df_work_viz.copy() # copy() 추가
                    else:
                        df_viz = df_work_viz[df_work_viz['topic_name'] == filter_topic_name].copy() # copy() 추가

                    # =======================================
                    # A. 주제별 성과 분포 (기존 유지)
                    # =======================================
                    st.subheader("A. 주제별 성과 분포")
                    try:
                        fig_topic_box = px.box(
                            df_viz, x='topic_name', y='engagement', color='topic_name',
                            title=f'주제(토픽)별 콘텐츠 매력 점수(Total Engagement) 분포 ({filter_topic_name})',
                            labels={'topic_name': '주제명', 'engagement': '콘텐츠 매력 점수(총합)'}
                        )
                        st.plotly_chart(fig_topic_box, use_container_width=True)
                    except Exception as e:
                        st.error(f"A. 토픽 성과 시각화 실패: {e}")

                    st.markdown("---")

                    # =======================================
                    # B. 감성(S/I)과 성과 (기존 C에서 승격)
                    # =======================================
                    st.subheader(f"B. 감성(S/I)과 성과 ({filter_topic_name})")
                    
                    if senti_pack_viz and senti_pack_viz.get('cv') and senti_pack_viz.get('lex'):
                        if 'S' not in df_viz.columns:
                            # compute_sentiment_SI는 원본 df를 복사하므로, 필터링된 df_viz에 다시 계산해야 함
                            df_viz = compute_sentiment_SI(df_viz, senti_pack_viz['cv'], senti_pack_viz['lex'])

                        if 'S' in df_viz.columns and df_viz['S'].abs().sum() > 0:
                            fig_senti_scatter = px.scatter(
                                df_viz, x='S', y='engagement', color='quality_label',
                                title=f'콘텐츠 감성(S)과 성과(Total Engagement) 관계 ({filter_topic_name})',
                                labels={'S': '평균 감성 점수 (S)', 'engagement': '콘텐츠 매력 점수(총합)'},
                                hover_data=['title'],
                                color_discrete_map={'good': 'blue', 'medium': 'gray', 'bad': 'red'}
                            )
                            st.plotly_chart(fig_senti_scatter, use_container_width=True)
                        else:
                            st.warning(f"'{filter_topic_name}' 토픽에서 유효한 감성 점수(S)를 계산할 수 없었습니다. (감성 사전 단어 부족)")
                    else:
                        st.caption("💖 '감성 사전 CSV'를 업로드하면 감성-성과 관계 분석이 활성화됩니다.")

                    st.markdown("---")

                    # =======================================
                    # C. 상위 N개 키워드 시계열 분석 (신규/교체)
                    # =======================================
                    st.subheader("C. 상위 N개 키워드 시계열 분석 (Good/Bad)")
                    st.caption("└ 'Good' 및 'Bad' 등급 콘텐츠에서 가장 빈번하게 등장하는 키워드의 시간 흐름에 따른 사용 빈도 변화를 분석합니다.")

                    # 선택 박스 추가
                    selected_term_type = st.radio("분석할 단어 유형", ["최근 인기 키워드", "전체 기간 성과 우수 키워드"], index=0, horizontal=True, key="c_term_type_select")
                    
                    k_top_words = st.slider("분석할 키워드 개수 (Top-K)", 5, 20, 10, step=1, key="k_top_words_c")

                    try:
                        df_all_data = st.session_state.get('df_for_analysis')
                        if df_all_data is None or df_all_data.empty or 'date' not in df_all_data.columns:
                            st.warning("데이터가 로드되지 않았거나 날짜('date') 컬럼이 없습니다. TAB2에서 CSV를 업로드하고 분석을 실행해주세요.")
                        else:
                            
                            top_keywords = []
                            if selected_term_type == "최근 인기 키워드":
                                # 가장 최근 날짜 기준으로 인기 키워드 추출 (30일 이내)
                                end_date = df_all_data['date'].max().date()
                                top_keywords = get_recent_popular_words(df_all_data, end_date, topic_id=None, k=k_top_words)
                            else: # 전체 기간 성과 우수 키워드
                                topic_term_bank = st.session_state.get("topic_term_bank")
                                if topic_term_bank:
                                    # 모든 토픽에서 'good' 단어를 합쳐서 상위 K개 선택
                                    all_good_terms = {}
                                    for topic_id, data in topic_term_bank.items():
                                        if data.get("status") == "ok":
                                            # score가 Log Ratio이므로 절댓값 대신 Log Ratio 값을 그대로 사용
                                            for term, score in data["good"]:
                                                all_good_terms[term] = all_good_terms.get(term, 0) + score
                                    top_keywords = sorted(all_good_terms.items(), key=lambda item: item[1], reverse=True)[:k_top_words]
                                    top_keywords = [term for term, score in top_keywords]
                                else:
                                    st.warning("토픽 단어 은행이 없습니다. '최근 인기 키워드'를 대신 사용합니다.")
                                    end_date = df_all_data['date'].max().date()
                                    top_keywords = get_recent_popular_words(df_all_data, end_date, topic_id=None, k=k_top_words)

                            if not top_keywords:
                                st.info("분석할 키워드를 찾지 못했습니다. TAB2에서 분석을 완료하거나 데이터셋을 확인해주세요.")
                            else:
                                st.markdown(f"**분석 키워드:** `{', '.join(top_keywords[:k_top_words])}`")

                                # 날짜를 월 단위로 그룹화
                                df_all_data['year_month'] = df_all_data['date'].dt.to_period('M')
                                
                                # 결과를 저장할 리스트
                                plot_data = []

                                # 각 키워드에 대해 Good/Bad 문서에서 월별 빈도 계산
                                for keyword in top_keywords[:k_top_words]:
                                    # 정규표현식 이스케이프 (특수문자 처리)
                                    escaped_keyword = regx.escape(keyword)

                                    # 'Good' 콘텐츠에서의 빈도
                                    df_good_monthly = df_all_data[df_all_data['quality_label'] == 'good'].groupby('year_month')['content'].apply(lambda x: x.str.contains(escaped_keyword, case=False).sum()).reset_index(name='count')
                                    df_good_monthly['keyword'] = keyword
                                    df_good_monthly['label'] = 'Good'
                                    plot_data.append(df_good_monthly)

                                    # 'Bad' 콘텐츠에서의 빈도
                                    df_bad_monthly = df_all_data[df_all_data['quality_label'] == 'bad'].groupby('year_month')['content'].apply(lambda x: x.str.contains(escaped_keyword, case=False).sum()).reset_index(name='count')
                                    df_bad_monthly['keyword'] = keyword
                                    df_bad_monthly['label'] = 'Bad'
                                    plot_data.append(df_bad_monthly)

                                if plot_data and not all(df.empty for df in plot_data):
                                    df_plot = pd.concat(plot_data)
                                    df_plot['year_month'] = df_plot['year_month'].dt.to_timestamp() # Plotly를 위해 datetime으로 변환
                                    
                                    # 키워드별로 그래프를 나누어 그립니다.
                                    fig_keyword_trend = px.line(
                                        df_plot, x='year_month', y='count', color='label', line_dash='keyword',
                                        title=f'키워드별 월간 사용 빈도 추이 (Good vs Bad)',
                                        labels={'year_month': '날짜', 'count': '월간 사용 빈도', 'label': '콘텐츠 등급', 'line_dash': '키워드'},
                                        color_discrete_map={'Good': 'blue', 'Bad': 'red'}
                                    )
                                    fig_keyword_trend.update_layout(hovermode="x unified")
                                    st.plotly_chart(fig_keyword_trend, use_container_width=True)
                                else:
                                    st.info("시계열 분석을 위한 데이터가 충분하지 않습니다.")

                    except Exception as e:
                        st.error(f"C. 키워드 시계열 분석 시각화 실패: {e}")

                    st.markdown("---")

                    # =======================================
                    # D. 핵심 피처 추세 (신규 - 기존 D 대체)
                    # =======================================
                    st.subheader("D. 핵심 피처 (길이/이미지) 구간별 성과 분석")
                    st.caption("└ 사용자가 직접 조정할 수 있는 피처(콘텐츠 길이, 이미지 수)의 변화가 성과에 미치는 영향을 등급별로 분석합니다.")
                    
                    feature_to_bin = st.selectbox("분석할 핵심 피처 선택", ["content_length", "title_length", "img_count"], index=0, key="bin_feature_select")
                    
                    try:
                        df_analysis = st.session_state.get('df_for_analysis').copy() # 원본에서 전체 사용

                        # 결측치 처리 및 5개 구간으로 나누기
                        df_analysis[feature_to_bin] = pd.to_numeric(df_analysis[feature_to_bin], errors='coerce').fillna(0)
                        
                        if df_analysis[feature_to_bin].max() == 0:
                            st.warning(f"선택된 피처 '{feature_to_bin}'의 값이 모두 0입니다. 분석을 건너뜁니다.")
                        else:
                            # 1. pd.cut을 사용하여 5개 구간으로 나누기
                            df_analysis['feature_bin'] = pd.cut(df_analysis[feature_to_bin], bins=5, include_lowest=True, duplicates='drop')
                            
                            # 2. [오류 수정]: Interval 객체를 문자열로 변환하여 JSON 직렬화 오류 방지
                            df_analysis['feature_bin'] = df_analysis['feature_bin'].astype(str) # <-- FIX: Interval to String
                            
                            # 구간별, 등급별 평균 Engagement 계산
                            df_trend = df_analysis.groupby(['feature_bin', 'quality_label'])['engagement'].mean().reset_index()
                            
                            # 플롯 생성
                            fig_bin_trend = px.bar(
                                df_trend, x='feature_bin', y='engagement', color='quality_label', barmode='group',
                                title=f'{feature_to_bin} 구간별 평균 콘텐츠 매력 점수(Engagement)',
                                labels={'feature_bin': f'{feature_to_bin} 구간', 'engagement': '평균 Engagement 점수', 'quality_label': '콘텐츠 등급'},
                                color_discrete_map={'good': 'blue', 'medium': 'gray', 'bad': 'red'}
                            )
                            fig_bin_trend.update_layout(xaxis={'categoryorder': 'category ascending'})
                            st.plotly_chart(fig_bin_trend, use_container_width=True)

                    except Exception as e:
                        st.error(f"D. 핵심 피처 추세 시각화 실패: {e}")

                    st.markdown("---")

                # --- 데이터 처리 오류 메시지 (유지) ---
            except Exception as e:
                st.error(f"파일 로드 또는 초기 전처리 오류: {e}") # ★★★ try-except 구문 추가 완료 ★★★
                st.session_state['analysis_done'] = False 
                st.session_state['df_for_analysis'] = None
                st.stop()
        else:
            # [수정 반영] CSV 파일 없을 때 친절한 안내 메시지 표시
            st.info("⬆️ **콘텐츠 및 성과 CSV 파일을 업로드하여 분석을 시작하세요.** \n\n파일이 준비되면 분석 버튼이 활성화됩니다.", icon="📝")


    # ================= TAB3 (모델 관리자) =================
    with TAB3_ADMIN:
        st.header("🔬 모델 관리자 (Admin)")
        st.info("이 탭은 TAB2에서 분석이 완료된 후 활성화됩니다. 현재 적용된 모델의 상태와 성능을 점검합니다.")

        # 세션에서 데이터 로드
        df_full = st.session_state.get('df_for_analysis')
        topic_bank = st.session_state.get('topic_term_bank')
        clf_pack_adv = st.session_state.get('clf_pack_adv')
        lda_model = st.session_state.get('lda_model')
        lda_vect = st.session_state.get('lda_vect')
        topic_labels = st.session_state.get('topic_labels', {})

        if not st.session_state.get('analysis_done', False) or df_full is None or df_full.empty or 'topic' not in df_full.columns or clf_pack_adv is None or lda_model is None or topic_bank is None:
            st.error("⚠️ 데이터가 없습니다. TAB2에서 먼저 '빠른 분석' 또는 '정밀 분석'을 실행해 주세요.")
        else:
            # --- 1. 불용어 ---
            st.subheader("1. 불용어(Stopwords) 관리")
            with st.expander("현재 적용 중인 기본 불용어 목록 보기"):
                st.text(f"총 {len(STOPWORDS_KO)}개 단어:")
                st.json(STOPWORDS_KO)

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
            st.subheader("2. 토픽별 핵심 단어 은행 (RandomForest/LogRatio 기반)") # 제목 수정
            st.markdown("""
            [정보] 이 단어 은행은 `build_topic_term_bank_rf_logratio` (RandomForest 중요도 + Log Ratio) 함수로 생성됩니다.
            - **성과 우수 단어 (Good):** 중요도가 높고 'Good' 콘텐츠에 상대적으로 더 자주 등장한 단어입니다. (추천)
            - **성과 저조 단어 (Bad):** 중요도가 높고 'Bad' 콘텐츠에 상대적으로 더 자주 등장한 단어입니다. (비권장)
            """)
            st.caption("└ Score는 Log Ratio 값이며, 절대값이 클수록 Good/Bad 콘텐츠 간의 사용 빈도 차이가 크다는 의미입니다.") # 설명 수정

            if topic_labels:
                topic_names_map = {v.get('name', k): int(k.split(' ')[1]) for k,v in topic_labels.items()}
                selected_name = st.selectbox("확인할 토픽 선택", list(topic_names_map.keys()))

                if selected_name:
                    selected_id = topic_names_map[selected_name]

                    if selected_id not in topic_bank:
                                st.error(f"토픽 {selected_id}가 단어 은행에 없습니다. (TAB2 재실행 필요)")
                    else:
                        bank_data = topic_bank[selected_id]
                        if bank_data.get("status") == "ok":
                            if bank_data.get("warning"):
                                st.warning(bank_data.get("warning"))

                            c_g, c_b, c_a = st.columns(3)
                            # Score (Log Ratio)도 함께 표시
                            c_g.dataframe({"성과 우수 단어 (Good)": [f"{w} ({s:.2f})" for w,s in bank_data['good'][:20]]})
                            c_b.dataframe({"성과 저조 단어 (Bad)": [f"{w} ({s:.2f})" for w,s in bank_data['bad'][:20]]})
                            c_a.dataframe({"단순 빈도 단어 (All)": [f"{w} ({s:.0f})" for w,s in bank_data['all'][:20]]})
                        else:
                            st.error(f"'{selected_name}' 토픽의 단어 은행을 표시할 수 없습니다.\n\n**사유:** {bank_data.get('message', '알 수 없는 오류')}")


            # --- 3. 모델 성능 평가 ---
            st.subheader("3. 모델 성능 평가")
            st.info("버튼을 누르면 Advanced Mode 피처셋과 Baseline Mode 피처셋에 대한 세 모델의 성능을 Stratified 5-Fold 교차 검증으로 비교합니다.")

            # [수정] 버튼 클릭 시 Baseline 평가 로직 추가
            if st.button("🚀 성능 평가 실행 (Baseline vs Advanced 비교)") or st.session_state.get('comparison_results_baseline') is None:
                with st.spinner("Advanced Mode 피처셋으로 3가지 분류기 모델을 학습 및 평가 중... (Stratified 5-Fold)"):
                    try:
                        # 1. Advanced Mode 평가 실행 (기존 로직)
                        adv_results = evaluate_comparison_models(df_full, lda_vect)
                        st.session_state['comparison_results_adv'] = adv_results
                        
                        # 2. Baseline Mode 평가 실행 (새로운 로직)
                        base_results = evaluate_baseline_models(df_full)
                        st.session_state['comparison_results_baseline'] = base_results

                    except Exception as e:
                        st.error(f"교차 검증 중 심각한 오류 발생: {e}")

            adv_results = st.session_state.get('comparison_results_adv')
            base_results = st.session_state.get('comparison_results_baseline')

            if adv_results and base_results:
                # B. Baseline/Advanced 통합 비교 테이블
                st.markdown("#### A. Baseline vs Advanced 모델 통합 성능 비교 (5-Fold 평균)")
                
                summary_data = []
                for name in ["SGDClassifier", "LogisticRegression", "RandomForestClassifier"]:
                    if name in adv_results and 'error' not in adv_results[name]:
                        # Advanced 결과
                        summary_data.append({
                            "모델": f"Advanced ({name})",
                            "Accuracy_Mean": adv_results[name]['Accuracy_Mean'],
                            "F1_Good_Mean": adv_results[name]['F1_Good_Mean']
                        })
                    if name in base_results and 'error' not in base_results[name]:
                        # Baseline 결과
                        summary_data.append({
                            "모델": f"Baseline ({name})",
                            "Accuracy_Mean": base_results[name]['Accuracy_Mean'],
                            "F1_Good_Mean": base_results[name]['F1_Good_Mean']
                        })

                if summary_data:
                    summary_df = pd.DataFrame(summary_data).set_index("모델").sort_values("F1_Good_Mean", ascending=False).round(3)
                    st.dataframe(summary_df, use_container_width=True)
                else:
                    st.warning("비교할 수 있는 유효한 모델 성능 결과가 없습니다.")


                # A. Advanced 모델 상세 평가 (기존 로직 유지)
                st.markdown("#### B. Advanced 모델 상세 평가 (Classification Report & CM)")
                for name, res in adv_results.items():
                    st.markdown(f"##### Advanced ({name}) 모델")
                    if 'error' in res:
                        st.error(res['error'])
                    else:
                        c_rep, c_cm = st.columns(2)
                        with c_rep:
                            st.text(f"Fold 평균 메트릭 (총 {res['Report_DF'].loc['N_Folds'].iloc[0]}개 Fold):")
                            st.dataframe(res['Report_DF'])
                        with c_cm:
                            st.text("Total Confusion Matrix (모든 Fold 합산):")
                            st.dataframe(pd.DataFrame(res['CM_Total'], index=['True: Good', 'True: Bad'], columns=['Pred: Good', 'Pred: Bad']))


    # ================= TAB4 (파인튜닝 관리자) =================
    with TAB4_FT:
        st.header("🤖 파인튜닝 관리자")
        st.info("이 탭은 LLM 제목 생성 모델(파인튜닝된 GPT)의 학습 데이터를 준비하고 새로운 모델 ID를 적용하는 곳입니다.")

        # --- 1. 현재 모델 상태 ---
        st.subheader("1. 현재 LLM 상태")
        st.markdown(f"**현재 사용 모델 ID:** `{FINETUNED_MODEL_ID_CURRENT}`")
        st.markdown(f"**현재 학습 작업 ID (Job ID):** `{st.session_state.get('ft_job_id', 'N/A')}`")
        
        if is_ft_model_ready:
            st.success("✅ 파인튜닝 모델이 활성화되어 있습니다.")
        else:
            st.warning("⚠️ 기본 모델 또는 더미 ID가 사용 중입니다. 파인튜닝을 통해 성능을 높일 수 있습니다.")
        st.markdown("---")


        # --- 2. 학습 데이터 생성 및 학습 시작 (자동화) ---
        st.subheader("2. 학습 데이터 생성 및 GPT 파인튜닝 시작")
        st.caption("TAB2에서 분석된 최신 'Good' 콘텐츠 패턴을 기반으로 학습 데이터를 만들고 OpenAI 학습 작업을 시작합니다.")
        
        # 기본 모델 선택
        base_model_ft = st.selectbox(
            "파인튜닝에 사용할 기본 모델", 
            options=["gpt-4o-mini-2024-07-18"], # gpt-4o-mini를 첫 번째 옵션으로 이동
            index=0, # [수정] gpt-4o-mini-2024-07-18 모델을 기본(index 0)으로 선택
            help="GPT-4o-mini는 파인튜닝 가능 여부가 자주 변경됩니다. 현재는 gpt-4o-mini를 기본으로 권장합니다."
        )

        is_analysis_ready = st.session_state.get('analysis_done', False)
        
        btn_start_ft = st.button("🔥 파인튜닝 학습 시작 (OpenAI API 호출)", disabled=not is_analysis_ready)

        if not is_analysis_ready:
            st.info("파인튜닝을 시작하려면 **TAB2에서 '정밀 분석'**을 먼저 완료해야 합니다.")


        if btn_start_ft:
            require_llm()
            df_full = st.session_state.get('df_for_analysis')
            topic_labels = st.session_state.get('topic_labels')

            # 이중 체크 (버튼 비활성화로 대부분 처리되지만 안전을 위해)
            if df_full is None or df_full.empty or topic_labels is None:
                st.error("⚠️ 파인튜닝을 시작하려면 TAB2에서 '정밀 분석'을 먼저 완료해야 합니다.")
            else:
                with st.spinner("LLM 학습 데이터셋 생성 및 OpenAI 학습 작업 시작 중..."):
                    try:
                        # run_finetuning_job 함수 호출 (analytics_core.py에 추가되어야 함)
                        # st.info/st.success 메시지는 run_finetuning_job 내부에 있습니다.
                        new_job_id = run_finetuning_job(df_full, topic_labels, base_model=base_model_ft)
                        st.session_state['ft_job_id'] = new_job_id
                        st.success(f"🎉 파인튜닝 작업 시작 완료! Job ID: `{new_job_id}`")
                        st.balloons()
                        st.info("파인튜닝은 수 시간이 소요될 수 있습니다. OpenAI 웹사이트에서 작업 상태를 확인하세요.")
                        st.rerun()
                    except APIError as e:
                        st.error(f"❌ OpenAI API 오류 (인증, 할당량 또는 데이터 문제): {e}")
                    except Exception as e:
                        st.error(f"❌ 파인튜닝 시작 중 예상치 못한 오류 발생: {e}")

        st.markdown("---")
        
        # --- 3. 모델 ID 적용 (파인튜닝 결과 반영) ---
        st.subheader("3. 학습 작업 상태 확인 및 모델 ID 적용")
        st.caption("파인튜닝 작업이 완료되면, OpenAI에서 발급받은 **새로운 모델 ID**를 여기에 입력하거나, 상태 확인 버튼을 눌러보세요.")
        
        # 학습 작업 상태 표시 및 업데이트 버튼
        job_id_current = st.session_state.get('ft_job_id')
        
        if job_id_current and LLM_OK:
            if st.button(f"🔄 학습 Job ID `{job_id_current[:15]}...` 상태 확인 및 ID 가져오기"):
                require_llm()
                with st.spinner(f"Job ID `{job_id_current}`의 상태를 조회 중..."):
                    try:
                        # OpenAI API 클라이언트 초기화는 analytics_core에서 이미 처리됨
                        # client 객체를 사용하여 조회
                        job_info = client.fine_tuning.jobs.retrieve(job_id_current)
                        status = job_info.status
                        
                        st.markdown(f"**현재 상태:** `{status}`")
                        
                        if status == 'succeeded' and job_info.fine_tuned_model:
                            new_ft_model_id = job_info.fine_tuned_model
                            st.session_state['ft_model_id'] = new_ft_model_id
                            st.success(f"✅ 학습 성공! 새로운 모델 ID `{new_ft_model_id}`가 적용되었습니다.")
                            st.balloons()
                            st.rerun()
                        elif status in ['running', 'queued']:
                            st.warning("⏳ 학습이 진행 중입니다. 잠시 후 다시 확인해주세요.")
                        elif status == 'failed':
                            st.error("❌ 학습이 실패했습니다. OpenAI 대시보드에서 원인을 확인하세요.")
                        
                    except APIError as e:
                        st.error(f"Job ID 조회 실패: {e}")
                    except Exception as e:
                        st.error(f"Job ID 조회 중 예상치 못한 오류: {e}")

        # 수동 모델 ID 입력 필드
        new_ft_model_id_input = st.text_input(
            "수동으로 파인튜닝 모델 ID 입력 (ft:gpt-3.5-turbo...)", 
            value=FINETUNED_MODEL_ID_CURRENT,
            key="ui_new_ft_model_id"
        )

        if st.button("✅ 새 모델 ID 수동 적용"):
            if new_ft_model_id_input.startswith("ft:"):
                st.session_state['ft_model_id'] = new_ft_model_id_input
                st.success(f"새 모델 ID `{new_ft_model_id_input[:30]}...`가 적용되었습니다. TAB1에서 확인하세요.")
                st.rerun()
            else:
                st.error("유효한 파인튜닝 모델 ID 형식(ft:...)을 입력하세요.")


if __name__ == '__main__':
    main()