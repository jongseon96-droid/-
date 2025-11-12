import pandas as pd
import os

# ==============================================================================
# 1. 설정 및 파일 경로 정의
# ==============================================================================

# [수정 필요 시] 분석할 파일들이 위치한 기본 경로를 지정합니다. (Windows 경로)
BASE_DIR = r"C:\Users\jongs\Downloads\open\legacy"

# 입력 파일명
CONTENT_FILE = "contents.csv"
METRICS_FILE = "article_metrics_monthly.csv"
IMAGES_FILE = "NumofImages.csv" 

# 출력 파일명
OUTPUT_FILE = "merged_analytics_data_total.csv" # 총합 데이터임을 명확히 하기 위해 파일명 변경

# 경로 조합
content_path = os.path.join(BASE_DIR, CONTENT_FILE)
metrics_path = os.path.join(BASE_DIR, METRICS_FILE)
images_path = os.path.join(BASE_DIR, IMAGES_FILE)
output_path = os.path.join(BASE_DIR, OUTPUT_FILE)

# ==============================================================================
# 2. 데이터 로드 및 전처리
# ==============================================================================

print("--- 데이터 병합 스크립트 실행 시작 (기사별 총합) ---")
print(f"기준 경로: {BASE_DIR}")

# 헬퍼 함수: CSV 파일을 로드합니다.
def load_data(file_path, df_name):
    """지정된 경로에서 CSV를 로드하고 에러 발생 시 처리합니다."""
    try:
        # 인코딩 문제 발생 시 'utf-8', 'cp949', 'euc-kr' 등을 시도해 보세요.
        df = pd.read_csv(file_path, encoding='utf-8')
        print(f"✅ {df_name} 로드 완료: {len(df)} 행")
        # 컬럼명을 소문자 및 언더스코어로 정규화 (병합 키 일관성 유지)
        df.columns = df.columns.str.lower().str.replace(' ', '_')
        if 'article_id' not in df.columns:
            raise KeyError(f"'article_id' 컬럼을 {df_name} 에서 찾을 수 없습니다.")
        return df
    except FileNotFoundError:
        print(f"❌ 오류: {df_name} 파일을 찾을 수 없습니다: {file_path}")
        return None
    except Exception as e:
        print(f"❌ 오류: {df_name} 로드 중 문제 발생 ({e.__class__.__name__}): {e}")
        return None


# 2.1. contents.csv 로드 및 글자 길이 계산
df_contents = load_data(content_path, CONTENT_FILE)
if df_contents is not None:
    # ✨ [수정] article_id를 문자열로 통일 (타입 불일치 오류 방지) ✨
    df_contents['article_id'] = df_contents['article_id'].astype(str)
    
    print("📝 contents 데이터 전처리 중: 글자 길이 계산 및 원본 텍스트 컬럼 유지...")
    df_contents['title_length'] = df_contents['title'].astype(str).apply(len)
    df_contents['content_length'] = df_contents['content'].astype(str).apply(len)

    # contents 파일에서 필요한 컬럼만 선택합니다. (date 컬럼이 있다고 가정)
    content_cols_to_keep = [
        'article_id', 'title', 'content', 'date', 'title_length', 'content_length'
    ]
    content_cols_to_keep = [col for col in content_cols_to_keep if col in df_contents.columns]

    df_contents_prep = df_contents[content_cols_to_keep].copy()
    base_df = df_contents_prep
else:
    print("⚠️ contents.csv 로드에 실패하여 병합을 시작할 수 없습니다.")
    exit()

# 2.2. article_metrics_monthly.csv 로드
df_metrics = load_data(metrics_path, METRICS_FILE)
if df_metrics is not None:
    # ✨ [수정] article_id를 문자열로 통일 (타입 불일치 오류 방지) ✨
    df_metrics['article_id'] = df_metrics['article_id'].astype(str)

# 2.3. NumofImages.scv 로드
df_images = load_data(images_path, IMAGES_FILE)
if df_images is not None:
    # ✨ [수정] article_id를 문자열로 통일 (타입 불일치 오류 방지) ✨
    df_images['article_id'] = df_images['article_id'].astype(str)


# ==============================================================================
# 3. 데이터 병합 (article_id 기준 총합)
# ==============================================================================

# 3.1. 기본 데이터 (contents)와 메트릭스 병합 (총합 계산)
if df_metrics is not None:
    print("🔗 contents 데이터와 metrics 데이터 병합 중 (article_id별 총합)...")
    
    metrics_cols = ['comments', 'likes', 'views_total'] # metrics only

    # [핵심 수정] article_id별로 metrics 컬럼을 총합(sum)하여 집계합니다.
    df_metrics_agg = df_metrics.groupby('article_id')[metrics_cols].sum().reset_index()

    # content features (base_df)에 총합 메트릭을 병합합니다.
    base_df = pd.merge(base_df, df_metrics_agg, on='article_id', how='left')
    
    # 병합 후 메트릭 컬럼이 NaN일 경우 0으로 채움 (metrics 기록이 없는 article_id)
    base_df[metrics_cols] = base_df[metrics_cols].fillna(0)
    print(f"✅ metrics 총합 병합 완료. 현재 행 수: {len(base_df)}")
    
    # period 컬럼은 총합 계산으로 인해 최종 base_df에 포함되지 않습니다.
else:
    print("❌ metrics 파일 로드 실패. 해당 컬럼들은 최종 파일에 포함되지 않습니다.")


# 3.2. 이미지 카운트 데이터 병합
if df_images is not None:
    print("🔗 현재 데이터와 NumofImages 데이터 병합 중...")
    
    img_cols = ['article_id', 'img_count']
    if all(col in df_images.columns for col in img_cols):
        # 이미지 수는 정적 정보이므로 article_id 기준으로 중복 제거
        df_images_unique = df_images.drop_duplicates(subset=['article_id'])[img_cols]
        
        # article_id 기준으로 left join하여 img_count 추가
        base_df = pd.merge(base_df, df_images_unique, on='article_id', how='left')
        base_df['img_count'] = base_df['img_count'].fillna(0)
        print(f"✅ NumofImages 병합 완료. 현재 행 수: {len(base_df)}")
    else:
        print(f"❌ NumofImages 파일에 'article_id' 또는 'img_count' 컬럼이 없어 병합을 건너뜁니다.")
else:
    print("❌ NumofImages 파일 로드 실패. 해당 컬럼은 최종 파일에 포함되지 않습니다.")


# 3.3. 최종 컬럼 정리 및 정렬
final_cols = [
    'article_id', 'title', 'content', 'date', 
    'views_total', 'likes', 'comments', 
    'title_length', 'content_length', 'img_count'
]
# 실제 base_df에 존재하는 컬럼만 선택
final_cols = [col for col in final_cols if col in base_df.columns]
base_df = base_df[final_cols]


# ==============================================================================
# 4. 결과 저장
# ==============================================================================

try:
    # 최종 결과 CSV 저장
    base_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print("\n")
    print("="*50)
    print(f"🎉 성공적으로 데이터 병합 및 저장 완료!")
    print(f"저장 경로: {output_path}")
    print(f"총 레코드 수: {len(base_df)}")
    print(f"최종 컬럼: {base_df.columns.tolist()}")
    print("="*50)

except Exception as e:
    print(f"\n❌ 최종 파일 저장 중 오류 발생: {e}")