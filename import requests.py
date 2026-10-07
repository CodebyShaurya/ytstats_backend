import requests
import json
import os
import re
import sys
from flask import Flask, jsonify, request

SERPAPI_API_KEY = os.getenv("SERPAPI_API_KEY")
app = Flask(__name__)


class SerpApiError(Exception):
    """Raised when SerpApi returns an API-level error."""


def extract_video_id(url):
    patterns = [
        r"(?:youtube\.com/watch\?v=)([a-zA-Z0-9_-]{11})",
        r"(?:youtu\.be/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/shorts/)([a-zA-Z0-9_-]{11})",
        r"(?:youtube\.com/embed/)([a-zA-Z0-9_-]{11})"
    ]

    for pattern in patterns:
        match = re.search(pattern, url)

        if match:
            return match.group(1)

    raise ValueError("Could not extract YouTube video ID")


def serpapi_request(params):
    if not SERPAPI_API_KEY:
        raise SerpApiError(
            "SERPAPI_API_KEY environment variable is not set."
        )

    params["api_key"] = SERPAPI_API_KEY

    response = requests.get(
        "https://serpapi.com/search",
        params=params,
        timeout=60
    )

    response.raise_for_status()

    data = response.json()

    if "error" in data:
        raise SerpApiError(data["error"])

    return data


# --------------------------------------------------
# VIDEO INFORMATION
# --------------------------------------------------

def get_video_info(video_id):

    params = {
        "engine": "youtube_video",
        "v": video_id
    }

    data = serpapi_request(params)

    return {
        "video_id": video_id,
        "title": data.get("title"),
        "description": data.get("description"),
        "views": data.get("extracted_views"),
        "likes": data.get("extracted_likes"),
        "comment_count": data.get("extracted_comment_count"),
        "published_date": data.get("published_date"),
        "duration": data.get("duration"),
        "channel": data.get("channel"),
        "thumbnail": data.get("thumbnail"),
        "url": f"https://www.youtube.com/watch?v={video_id}"
    }


# --------------------------------------------------
# COMMENTS
# --------------------------------------------------

def get_comments(video_id):

    comments = []

    params = {
        "engine": "youtube_video",
        "v": video_id
    }

    while True:

        data = serpapi_request(params)

        page_comments = data.get("comments", [])

        for comment in page_comments:

            comments.append({
                "comment_id": comment.get("comment_id"),

                "author": (
                    comment.get("channel", {}).get("name")
                    if isinstance(comment.get("channel"), dict)
                    else None
                ),

                "text": comment.get("snippet"),

                "likes": comment.get("likes"),

                "published": comment.get("published"),

                "replies_count": comment.get("replies_count")
            })

        next_token = data.get("comments_next_page_token")

        if not next_token:
            break

        params["next_page_token"] = next_token

        print(f"Fetched {len(comments)} comments...")

    return comments


# --------------------------------------------------
# TRANSCRIPT
# --------------------------------------------------

def get_transcript(video_id):

    params = {
        "engine": "youtube_video_transcript",
        "v": video_id
    }

    data = serpapi_request(params)

    transcript = []

    for item in data.get("transcript", []):

        transcript.append({
            "start": item.get("start_time_text"),
            "start_ms": item.get("start_ms"),
            "end_ms": item.get("end_ms"),
            "text": item.get("snippet")
        })

    return transcript


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def scrape_youtube_video(video_url):

    print("\nExtracting video ID...")

    video_id = extract_video_id(video_url)

    print(f"Video ID: {video_id}")

    # Video information
    print("\nGetting video information...")

    video_info = get_video_info(video_id)

    # Transcript
    print("Getting transcript...")

    transcript = get_transcript(video_id)

    # Comments
    print("Getting comments...")

    comments = get_comments(video_id)

    result = {
        "video": video_info,
        "transcript": transcript,
        "comments": comments
    }

    return result


# --------------------------------------------------
# SAVE
# --------------------------------------------------

def save_results(data):

    with open(
        "youtube_data.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\nSaved to: youtube_data.json")


# --------------------------------------------------
# FLASK API
# --------------------------------------------------

@app.route("/api/youtube", methods=["GET", "POST"])
def youtube_api():
    """Return video information, transcript, description, and full comments."""
    if request.method == "POST":
        payload = request.get_json(silent=True) or {}
        video_url = payload.get("video_url") or payload.get("url")
    else:
        video_url = request.args.get("video_url") or request.args.get("url")

    if not video_url:
        return jsonify({
            "error": "video_url is required",
            "example": {
                "GET": "/api/youtube?video_url=https://www.youtube.com/watch?v=VIDEO_ID",
                "POST": {"video_url": "https://www.youtube.com/watch?v=VIDEO_ID"}
            }
        }), 400

    try:
        data = scrape_youtube_video(video_url)
        return jsonify(data)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except SerpApiError as error:
        return jsonify({"error": str(error)}), 502
    except requests.RequestException as error:
        return jsonify({"error": f"SerpApi request failed: {error}"}), 502


def print_results(data):
    """Print the returned description and each fetched comment."""
    print("\n" + "=" * 50)
    print("SCRAPING COMPLETE")
    print("=" * 50)
    print(f"Title: {data['video']['title']}")
    print(f"Views: {data['video']['views']}")
    print(f"Likes: {data['video']['likes']}")
    print(f"Comments: {data['video']['comment_count']}")
    print(f"Comments fetched: {len(data['comments'])}")
    print(f"Transcript segments: {len(data['transcript'])}")
    print("\nDescription:")
    print(data["video"]["description"] or "(No description)")
    print("\nComments:")
    if not data["comments"]:
        print("(No comments found)")
        return

    for index, comment in enumerate(data["comments"], start=1):
        print(f"\n[{index}] {comment.get('author') or 'Unknown author'}")
        print(comment.get("text") or "(No comment text)")
        print(f"Likes: {comment.get('likes')}")
        print(f"Published: {comment.get('published')}")


# --------------------------------------------------
# RUN
# --------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) > 1 and sys.argv[1] == "api":
        app.run(host="0.0.0.0", port=5000, debug=False)
    elif len(sys.argv) > 1:

        video_url = sys.argv[1]

    else:

        video_url = input(
            "Enter YouTube video URL: "
        )

    try:

        data = scrape_youtube_video(video_url)

        save_results(data)
        print_results(data)

    except Exception as e:

        print(f"\nERROR: {e}")