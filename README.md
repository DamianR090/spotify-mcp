# Spotify MCP (remote, cloud-hosted)

A self-contained MCP server that gives Claude **full control of your Spotify
account** — running on a cloud box, completely separate from your computer. Your
authorization lives on the server, not your machine.

It does the things the built-in Spotify connector can't, most importantly
**reading the actual tracks inside a playlist and your Liked Songs**, plus top
tracks/artists, recently played, followed artists, full catalog search, and
every playlist write operation. 24 tools total.

---

## ⚠️ Read this first (honest expectations)

- **This does not retroactively upgrade an existing chat.** A Claude
  conversation's connectors are fixed when it starts. Once you connect this
  server, **new** chats get the full toolset.
- **You host it** (a few minutes on a free tier). It can't be hosted for you.
- Spotify killed **Recommendations / Related Artists / Audio Features** for apps
  created after Nov 27 2024, so the "find similar" tool reconstructs similarity
  with **Last.fm** (optional, free key) or a genre-search fallback.
- New Spotify apps run in **development mode** (personal use, up to 25 users) —
  perfect for this, no review needed.

---

## What you need

1. A free **Spotify Developer** app — https://developer.spotify.com/dashboard
2. A free cloud host (this guide uses **Render**; Railway/Fly work too)
3. *(Optional)* a free **Last.fm API key** for real similarity —
   https://www.last.fm/api/account/create

---

## Setup (≈10 minutes)

### 1. Create the Spotify app
In the dashboard → **Create app**. Note the **Client ID** and **Client Secret**.
Leave Redirect URIs for now — you'll add the real one in step 3.

### 2. Deploy the server
Push these files to a GitHub repo, then on Render → **New → Web Service** →
point it at the repo. The included `render.yaml` sets it up automatically
(build: `pip install -r requirements.txt`, start: `python app.py`).

Set these environment variables (leave `SPOTIFY_REFRESH_TOKEN` blank for now):

| Variable | Value |
|---|---|
| `SPOTIFY_CLIENT_ID` | from step 1 |
| `SPOTIFY_CLIENT_SECRET` | from step 1 |
| `SPOTIFY_REDIRECT_URI` | `https://<your-service>.onrender.com/callback` |
| `LASTFM_API_KEY` | *(optional)* |

Deploy. Confirm it's live by visiting `https://<your-service>.onrender.com/health`.

### 3. Register the redirect URI
Back in the Spotify dashboard → **Settings → Redirect URIs**, add the **exact**
value you used for `SPOTIFY_REDIRECT_URI` (e.g.
`https://<your-service>.onrender.com/callback`) and save.

### 4. Authorize once
Visit `https://<your-service>.onrender.com/login` in a browser, approve the
Spotify permissions. The callback page prints a **refresh token**.
Copy it → set it as the `SPOTIFY_REFRESH_TOKEN` env var on Render → redeploy.
(This is what lets the server survive restarts.) Re-check `/health`; it should
now show `"authorized": true`.

### 5. Connect to Claude
Your MCP endpoint is: **`https://<your-service>.onrender.com/mcp`**

- **Claude.ai (web/mobile):** Settings → Connectors → add a custom connector →
  paste the `/mcp` URL.
- **Claude Desktop / Claude Code / OpenClaw:** add to your MCP config, e.g.
  ```json
  {
    "mcpServers": {
      "spotify": { "url": "https://<your-service>.onrender.com/mcp" }
    }
  }
  ```

Start a **new** chat and ask Claude to, e.g., "list the tracks in my Locked in
playlist" or "find 20 songs similar to my top artist and make a playlist."

---

## Security notes

- The refresh token grants control of your Spotify account — treat it like a
  password. It's stored only as a server env var.
- This single-user build doesn't put auth in front of the `/mcp` endpoint, so
  **keep your server URL private**. Anyone who learns it could drive your
  Spotify. For stronger isolation, put the host behind access controls or use a
  client that can send an auth header.
- Nothing here touches or runs on your personal computer.

---

## Heads-up on free tiers

Render's free web services **sleep after ~15 min idle** and cold-start in
30–60s, so the first Claude call after a quiet period may be slow or time out —
just retry. For always-on, use a paid instance or a host like Fly/Railway.

---

## Tools (24)

**Reads:** `spotify_get_me`, `spotify_search`, `spotify_get_liked_songs`,
`spotify_get_saved_albums`, `spotify_get_top_items`,
`spotify_get_recently_played`, `spotify_get_followed_artists`,
`spotify_list_playlists`, `spotify_get_playlist`, `spotify_get_playlist_tracks`,
`spotify_get_tracks`, `spotify_get_artists`, `spotify_get_artist_albums`,
`spotify_get_album_tracks`, `spotify_get_currently_playing`,
`spotify_find_similar`

**Writes:** `spotify_create_playlist`, `spotify_add_tracks_to_playlist`,
`spotify_remove_tracks_from_playlist`, `spotify_change_playlist_details`,
`spotify_save_tracks`, `spotify_remove_saved_tracks`, `spotify_follow_artists`,
`spotify_unfollow_artists`

---

## Local testing (optional)

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in values; use http://127.0.0.1:8000/callback
set -a; source .env; set +a
python app.py
# then visit http://127.0.0.1:8000/login
```
Note: Spotify no longer allows `localhost` as a redirect host — use `127.0.0.1`.
