import re
from collections import Counter

import streamlit as st
import pandas as pd
import numpy as np

from googleapiclient.discovery import build

from kiwipiepy import Kiwi

from wordcloud import WordCloud

import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

import plotly.express as px


# ---------------------------
# 기본 설정
# ---------------------------

st.set_page_config(
    page_title="YouTube 댓글 분석기",
    layout="wide"
)

st.title("🎬 YouTube 댓글 심층 분석기")

api_key = st.secrets["YOUTUBE_API_KEY"]

youtube = build(
    "youtube",
    "v3",
    developerKey=api_key
)


# ---------------------------
# 영상 ID 추출
# ---------------------------

def extract_video_id(url):

    patterns = [
        r"v=([a-zA-Z0-9_-]{11})",
        r"youtu\.be\/([a-zA-Z0-9_-]{11})",
        r"shorts\/([a-zA-Z0-9_-]{11})"
    ]

    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)

    return None


# ---------------------------
# 댓글 수집
# ---------------------------

def get_comments(video_id, max_comments=1000):

    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )

    while request and len(comments) < max_comments:

        response = request.execute()

        for item in response["items"]:

            comment = item["snippet"]["topLevelComment"]["snippet"]

            comments.append({
                "author": comment["authorDisplayName"],
                "text": comment["textDisplay"],
                "likes": comment["likeCount"]
            })

        request = youtube.commentThreads().list_next(
            request,
            response
        )

    return pd.DataFrame(comments)


# ---------------------------
# 명사 추출
# ---------------------------

kiwi = Kiwi()

stopwords = {
    "영상","진짜","너무","정말","그냥","이거",
    "저는","제가","입니다","있습니다",
    "그리고","하는","같은","에서","으로"
}

def extract_keywords(texts):

    nouns = []

    for text in texts:

        tokens = kiwi.tokenize(str(text))

        for token in tokens:

            if token.tag.startswith("N"):

                word = token.form

                if len(word) >= 2 and word not in stopwords:
                    nouns.append(word)

    return nouns


# ---------------------------
# 워드클라우드
# ---------------------------

def create_wordcloud(words):

    text = " ".join(words)

    wc = WordCloud(
        font_path="fonts/NanumGothic.ttf",
        width=1200,
        height=700,
        background_color="white"
    )

    return wc.generate(text)


# ---------------------------
# UI
# ---------------------------

url = st.text_input(
    "유튜브 링크 입력"
)

if st.button("분석 시작"):

    video_id = extract_video_id(url)

    if not video_id:
        st.error("올바른 유튜브 URL이 아닙니다.")
        st.stop()

    with st.spinner("댓글 수집 중..."):

        df = get_comments(video_id)

    if len(df) == 0:
        st.error("댓글을 가져오지 못했습니다.")
        st.stop()

    st.success(f"{len(df):,}개 댓글 수집 완료")

    # ---------------------------
    # 통계
    # ---------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("댓글 수", f"{len(df):,}")

    col2.metric(
        "작성자 수",
        df["author"].nunique()
    )

    col3.metric(
        "평균 좋아요",
        round(df["likes"].mean(), 1)
    )

    col4.metric(
        "평균 댓글 길이",
        round(df["text"].str.len().mean(), 1)
    )

    # ---------------------------
    # 키워드 분석
    # ---------------------------

    st.header("🔥 핵심 키워드")

    words = extract_keywords(df["text"])

    top_keywords = Counter(words).most_common(30)

    keyword_df = pd.DataFrame(
        top_keywords,
        columns=["키워드", "빈도"]
    )

    fig = px.bar(
        keyword_df,
        x="키워드",
        y="빈도"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # ---------------------------
    # 워드클라우드
    # ---------------------------

    st.header("☁️ 워드클라우드")

    wc = create_wordcloud(words)

    fig_wc, ax = plt.subplots(
        figsize=(12, 7)
    )

    ax.imshow(wc)

    ax.axis("off")

    st.pyplot(fig_wc)

    # ---------------------------
    # 댓글 클러스터링
    # ---------------------------

    st.header("🧠 댓글 주제 분석")

    texts = df["text"].astype(str)

    vectorizer = TfidfVectorizer(
        max_features=1000
    )

    X = vectorizer.fit_transform(texts)

    n_clusters = min(5, len(df))

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )

    labels = model.fit_predict(X)

    df["cluster"] = labels

    terms = vectorizer.get_feature_names_out()

    for cluster_id in range(n_clusters):

        st.subheader(
            f"주제 그룹 {cluster_id+1}"
        )

        center = model.cluster_centers_[cluster_id]

        top_indices = center.argsort()[-10:][::-1]

        keywords = [
            terms[i]
            for i in top_indices
        ]

        st.write(
            "대표 키워드:",
            ", ".join(keywords)
        )

        sample = (
            df[df["cluster"] == cluster_id]
            .head(5)["text"]
        )

        for s in sample:
            st.write("•", s)

    # ---------------------------
    # 좋아요 TOP 댓글
    # ---------------------------

    st.header("🏆 좋아요 TOP 댓글")

    top_comments = (
        df.sort_values(
            "likes",
            ascending=False
        )
        .head(20)
    )

    st.dataframe(
        top_comments,
        use_container_width=True
    )
