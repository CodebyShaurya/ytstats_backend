# ytstats_backend

The `/api/youtube` endpoint accepts a YouTube URL and sends a user-selected
comment sample, video metadata, and the transcript to Gemini. It returns only
the `analysis` object and the number of comments analyzed; raw comments are
not returned.

Use either an explicit `comment_count` (between 1 and 500) or an accuracy
preset:

- `low`: 50 comments;
- `medium`: 200 comments;
- `high`: 500 comments.

Example:

```text
GET /api/youtube?yturl=https://www.youtube.com/watch?v=VIDEO_ID&accuracy=high
```

The response includes:

- the general viewer consensus and sentiment;
- title and description match scores from 0 to 10;
- an engagement assessment using likes and comment counts; and
- an overall rating from 0 to 10 with an explanation and limitations;
- up to three related videos, each with a YouTube link; and
- a `lacking_topic_videos` list containing one video link for each topic
  Gemini identifies as inadequately covered.

Set both `SERPAPI_API_KEY` and `GEMINI_API_KEY` in `.env`. `GEMINI_MODEL` is
optional and defaults to `gemini-2.5-flash`. Preview, experimental, or
unsupported configured model names are ignored, and the API discovers a
stable available model that supports `generateContent`.
