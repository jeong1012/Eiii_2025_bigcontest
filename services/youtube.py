# services/youtube.py
import os
from googleapiclient.discovery import build
from datetime import datetime, timedelta

def search_videos_by_query(query: str, max_results: int = 3):
    """
    주어진 검색어로 유튜브 영상을 검색하되, 결과가 없으면 키워드를 하나씩 줄여가며
    단계적으로 재검색하는 기능이 추가된 함수.
    ✨ 조회수 1만 회 이상 필터링, 관련성 높은 순 정렬 기능으로 개선
    """
    if not query:
        return "검색어가 제공되지 않았습니다.", []

    print(f"🔎 원본 검색어: '{query}'")

    YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
    if not YOUTUBE_API_KEY:
        raise ValueError("YOUTUBE_API_KEY 환경변수를 설정해주세요.")
        
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # 최근 5년 내 영상만 검색하도록 날짜 필터 설정
        five_years_ago = datetime.now() - timedelta(days=365*5)
        published_after_date = five_years_ago.strftime('%Y-%m-%dT%H:%M:%SZ')
        
        # ✨ 1. 최소 조회수와 후보군 개수 설정
        MIN_VIEW_COUNT = 10000
        CANDIDATES_TO_FETCH = 20 # 필터링을 위해 더 많은 후보군을 요청

        keywords = query.split()

        for i in range(len(keywords), 0, -1):
            current_query = " ".join(keywords[:i])
            print(f"▶️ '{current_query}' (으)로 유튜브 검색 시도...")

            # ✨ 2. 1단계: 관련성 순으로 후보 영상 ID 목록 확보
            search_request = youtube.search().list(
                q=current_query,
                part="id",
                type="video",
                order="relevance", # 관련성 높은 순으로 정렬
                maxResults=CANDIDATES_TO_FETCH,
                publishedAfter=published_after_date
            )
            search_response = search_request.execute()
            
            video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]

            if not video_ids:
                continue # 후보 영상이 없으면 다음 키워드로 넘어감

            # ✨ 3. 2단계: 후보 영상들의 상세 정보(조회수 포함) 요청
            videos_request = youtube.videos().list(
                part="snippet,statistics",
                id=",".join(video_ids)
            )
            videos_response = videos_request.execute()

            # ✨ 4. 필터링 및 최종 결과 선택
            final_videos = []
            for item in videos_response.get("items", []):
                view_count = int(item["statistics"].get("viewCount", 0))
                
                # 조회수 1만 회 이상인 영상만 최종 결과에 추가
                if view_count >= MIN_VIEW_COUNT:
                    video_id = item["id"]
                    title = item["snippet"]["title"]
                    link = f"https://www.youtube.com/watch?v={video_id}"
                    final_videos.append((title, link))
                    
                    # 원하는 개수(max_results)만큼 찾으면 중단
                    if len(final_videos) == max_results:
                        break
            
            if final_videos:
                print(f"✅ 결과 찾음! (조회수 {MIN_VIEW_COUNT}회 이상 필터링)")
                log_msg = f"'{current_query}' 검색 결과입니다."
                if current_query != query:
                    log_msg = f"'{query}'에 대한 결과가 없어, '{current_query}'(으)로 다시 검색한 결과입니다."
                
                return log_msg, final_videos

        return f"'{query}'에 대한 관련 영상을 찾지 못했습니다.", []

    except Exception as e:
        print(f"### 유튜브 API 오류: {e}")
        return "유튜브 검색 중 오류가 발생했습니다.", []