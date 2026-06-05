"""
spotify_mcp — a remote, cloud-hosted MCP server for full Spotify control.

This server is designed to run on a cloud host (Render, Railway, Fly, etc.),
completely separate from your personal machine. It holds your Spotify
authorization server-side and exposes it to Claude as a custom connector over
streamable HTTP.

It gives Claude the things Spotify still allows (post Nov-2024 / Feb-2026 API
changes): full library + playlist *reads* (including the actual tracks inside a
playlist and your liked songs), top tracks/artists, recently played, followed
artists, saved albums, full catalog search, and every playlist write operation.

Spotify killed Recommendations / Related Artists / Audio Features for new apps,
so `spotify_find_similar` reimplements similarity with an optional Last.fm hook
(set LASTFM_API_KEY) and a genre+catalog-search fallback.

Endpoints exposed by this process:
  GET  /health            -> liveness probe + auth status
  GET  /login             -> start the one-time Spotify OAuth (open in a browser)
  GET  /callback          -> OAuth redirect target; prints your refresh token
  POST /mcp               -> the MCP streamable-HTTP endpoint (use this in Claude)
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import time
import urllib.parse
from enum import Enum
from typing import Any, Dict, List, Optional

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, ConfigDict, Field
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse

# --------------------------------------------------------------------------- #
# Constants & configuration
# --------------------------------------------------------------------------- #

SPOTIFY_API_BASE = "https://api.spotify.com/v1"
SPOTIFY_ACCOUNTS = "https://accounts.spotify.com"
LASTFM_API_BASE = "https://ws.audioscrobbler.com/2.0/"
HTTP_TIMEOUT = 30.0
TOKEN_REFRESH_BUFFER = 60  # refresh this many seconds before expiry

# Scopes: everything this server's tools can need. The user grants these once.
SCOPES = " ".join([
    "user-read-private",
    "user-read-email",
    "user-library-read",
    "user-library-modify",
    "playlist-read-private",
    "playlist-read-collaborative",
    "playlist-modify-private",
    "playlist-modify-public",
    "user-top-read",
    "user-read-recently-played",
    "user-follow-read",
    "user-follow-modify",
    "user-read-playback-state",
    "user-read-currently-playing",
])

CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "")
# Must EXACTLY match a Redirect URI registered in your Spotify app dashboard.
REDIRECT_URI = os.environ.get("SPOTIFY_REDIRECT_URI", "")
LASTFM_API_KEY = os.environ.get("LASTFM_API_KEY", "")

# In-memory token state. Seeded from SPOTIFY_REFRESH_TOKEN env var on boot so the
# server survives restarts on hosts without a persistent disk.
_token_state: Dict[str, Any] = {
    "access_token": None,
    "refresh_token": os.environ.get("SPOTIFY_REFRESH_TOKEN") or None,
    "expires_at": 0.0,
}
# CSRF state values issued by /login and consumed by /callback.
_oauth_states: set[str] = set()


# --------------------------------------------------------------------------- #
# Spotify HTTP client (auth + request plumbing, shared by all tools)
# --------------------------------------------------------------------------- #

class SpotifyError(Exception):
    """Raised for actionable, user-facing failures."""


def _basic_auth_header() -> str:
    raw = f"{CLIENT_ID}:{CLIENT_SECRET}".encode()
    return "Basic " + base64.b64encode(raw).decode()


async def _refresh_access_token() -> None:
    """Exchange the stored refresh token for a fresh access token."""
    refresh_token = _token_state.get("refresh_token")
    if not refresh_token:
        raise SpotifyError(
            "Not authorized yet. Open the server's /login URL in a browser, "
            "complete Spotify login, then set SPOTIFY_REFRESH_TOKEN on the host."
        )
    if not (CLIENT_ID and CLIENT_SECRET):
        raise SpotifyError(
            "Server misconfigured: SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET are not set."
        )
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(
            f"{SPOTIFY_ACCOUNTS}/api/token",
            headers={"Authorization": _basic_auth_header(),
                     "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "refresh_token", "refresh_token": refresh_token},
        )
    if resp.status_code != 200:
        raise SpotifyError(
            f"Token refresh failed ({resp.status_code}). The refresh token may be "
            f"revoked or invalid — re-run /login. Detail: {resp.text[:200]}"
        )
    payload = resp.json()
    _token_state["access_token"] = payload["access_token"]
    _token_state["expires_at"] = time.time() + int(payload.get("expires_in", 3600))
    # Spotify may rotate the refresh token; keep the newest if returned.
    if payload.get("refresh_token"):
        _token_state["refresh_token"] = payload["refresh_token"]


async def _ensure_token() -> str:
    if (not _token_state.get("access_token")
            or time.time() >= _token_state.get("expires_at", 0) - TOKEN_REFRESH_BUFFER):
        await _refresh_access_token()
    return _token_state["access_token"]


async def _api(method: str, path: str, *, params: Optional[dict] = None,
               json_body: Optional[dict] = None, _retry: bool = True) -> Any:
    """Authenticated Spotify Web API request. `path` is relative to /v1.

    Returns parsed JSON, or {} for empty 2xx bodies. Raises SpotifyError with an
    actionable message on failure.
    """
    token = await _ensure_token()
    url = path if path.startswith("http") else f"{SPOTIFY_API_BASE}{path}"
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.request(
            method, url,
            headers={"Authorization": f"Bearer {token}",
                     "Content-Type": "application/json"},
            params=params, json=json_body,
        )

    if resp.status_code == 401 and _retry:
        # Access token expired mid-flight — force one refresh and retry once.
        _token_state["access_token"] = None
        return await _api(method, path, params=params, json_body=json_body, _retry=False)
    if resp.status_code == 429:
        retry_after = resp.headers.get("Retry-After", "a few")
        raise SpotifyError(f"Rate limited by Spotify. Retry after {retry_after} seconds.")
    if resp.status_code == 403:
        raise SpotifyError(
            f"Spotify returned 403 Forbidden. Spotify's actual response: {resp.text[:400]} "
            f"| Request was: {method} {url} params={params}"
        )
    if resp.status_code == 404:
        raise SpotifyError("Not found (404). Double-check the id/URI you passed.")
    if resp.status_code >= 400:
        raise SpotifyError(f"Spotify API error {resp.status_code}: {resp.text[:300]}")

    if resp.status_code == 204 or not resp.content:
        return {}
    return resp.json()


async def _paginate(path: str, *, params: Optional[dict] = None,
                    item_limit: int = 200, page_size: int = 50,
                    items_key: str = "items") -> List[dict]:
    """Follow Spotify paging objects until `item_limit` items are collected."""
    params = dict(params or {})
    params["limit"] = min(page_size, item_limit)
    collected: List[dict] = []
    next_url: Optional[str] = None
    while True:
        data = await _api("GET", next_url or path, params=None if next_url else params)
        page = data.get(items_key, data.get("items", []))
        collected.extend(page)
        if len(collected) >= item_limit:
            return collected[:item_limit]
        next_url = data.get("next")
        if not next_url:
            return collected


# --------------------------------------------------------------------------- #
# Compact formatters — keep tool responses small so they don't flood context
# --------------------------------------------------------------------------- #

def _artists_str(item: dict) -> str:
    return ", ".join(a.get("name", "") for a in item.get("artists", []))


def _fmt_track(t: dict) -> dict:
    if t and t.get("track"):  # playlist item wrapper
        t = t["track"]
    if not t:
        return {}
    return {
        "name": t.get("name"),
        "artists": _artists_str(t),
        "album": (t.get("album") or {}).get("name"),
        "id": t.get("id"),
        "uri": t.get("uri"),
        "duration_ms": t.get("duration_ms"),
        "popularity": t.get("popularity"),
    }


def _fmt_artist(a: dict) -> dict:
    return {
        "name": a.get("name"),
        "id": a.get("id"),
        "uri": a.get("uri"),
        "genres": a.get("genres", []),
        "followers": (a.get("followers") or {}).get("total"),
        "popularity": a.get("popularity"),
    }


def _fmt_album(a: dict) -> dict:
    return {
        "name": a.get("name"),
        "artists": _artists_str(a),
        "id": a.get("id"),
        "uri": a.get("uri"),
        "release_date": a.get("release_date"),
        "total_tracks": a.get("total_tracks"),
    }


def _fmt_playlist(p: dict) -> dict:
    return {
        "name": p.get("name"),
        "id": p.get("id"),
        "uri": p.get("uri"),
        "owner": (p.get("owner") or {}).get("display_name"),
        "tracks": (p.get("tracks") or {}).get("total"),
        "public": p.get("public"),
        "description": p.get("description"),
    }


def _ok(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def _err(e: Exception) -> str:
    msg = str(e) if isinstance(e, SpotifyError) else f"{type(e).__name__}: {e}"
    return json.dumps({"error": msg}, indent=2)


def _to_uris(ids_or_uris: List[str]) -> List[str]:
    """Accept bare track ids OR full spotify:track: URIs; normalize to URIs."""
    out = []
    for x in ids_or_uris:
        x = x.strip()
        out.append(x if x.startswith("spotify:") else f"spotify:track:{x}")
    return out


# --------------------------------------------------------------------------- #
# MCP server
# --------------------------------------------------------------------------- #

mcp = FastMCP(
    "spotify_mcp",
    stateless_http=True,
    # The MCP layer's DNS-rebinding guard otherwise 421s requests whose Host
    # isn't on a localhost allowlist (e.g. the Render domain). Our security
    # boundary is the OAuth token + private URL, so disable the host check.
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

RO = {"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True}
WRITE = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True}
DESTRUCTIVE = {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True}

_CFG = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra="forbid")


class TimeRange(str, Enum):
    short = "short_term"   # ~4 weeks
    medium = "medium_term"  # ~6 months
    long = "long_term"     # several years


class TopType(str, Enum):
    artists = "artists"
    tracks = "tracks"


class SearchType(str, Enum):
    track = "track"
    artist = "artist"
    album = "album"
    playlist = "playlist"


# ----- Profile & search ---------------------------------------------------- #

class Empty(BaseModel):
    model_config = _CFG


@mcp.tool(name="spotify_get_me", annotations={"title": "Get my Spotify profile", **RO})
async def spotify_get_me(params: Empty) -> str:
    """Return the authorized user's Spotify profile (display name, id, country, product tier, followers)."""
    try:
        me = await _api("GET", "/me")
        return _ok({"display_name": me.get("display_name"), "id": me.get("id"),
                    "country": me.get("country"), "product": me.get("product"),
                    "followers": (me.get("followers") or {}).get("total"),
                    "uri": me.get("uri")})
    except Exception as e:
        return _err(e)


class SearchInput(BaseModel):
    model_config = _CFG
    query: str = Field(..., description="Search text. Supports field filters like 'artist:', 'year:2018-2024', 'genre:'.", min_length=1)
    types: List[SearchType] = Field(default_factory=lambda: [SearchType.track],
                                    description="What to search for.", max_length=4)
    limit: int = Field(default=10, description="Max results per type (1-50).", ge=1, le=50)
    market: Optional[str] = Field(default=None, description="ISO country code, e.g. 'CA'. Omit to use the account's market.", max_length=2)


@mcp.tool(name="spotify_search", annotations={"title": "Search the Spotify catalog", **RO})
async def spotify_search(params: SearchInput) -> str:
    """Search Spotify's catalog for tracks, artists, albums, and/or playlists.

    Returns: JSON keyed by type, each a list of compact objects with name, id, and uri.
    """
    try:
        p = {"q": params.query, "type": ",".join(t.value for t in params.types), "limit": params.limit}
        if params.market:
            p["market"] = params.market
        data = await _api("GET", "/search", params=p)
        out: Dict[str, Any] = {}
        if "tracks" in data:
            out["tracks"] = [_fmt_track(t) for t in data["tracks"]["items"]]
        if "artists" in data:
            out["artists"] = [_fmt_artist(a) for a in data["artists"]["items"]]
        if "albums" in data:
            out["albums"] = [_fmt_album(a) for a in data["albums"]["items"]]
        if "playlists" in data:
            out["playlists"] = [_fmt_playlist(pl) for pl in data["playlists"]["items"] if pl]
        return _ok(out)
    except Exception as e:
        return _err(e)


# ----- Library reads ------------------------------------------------------- #

class LikedSongsInput(BaseModel):
    model_config = _CFG
    limit: int = Field(default=50, description="How many liked songs to fetch, newest first (1-1000).", ge=1, le=1000)


@mcp.tool(name="spotify_get_liked_songs", annotations={"title": "Get my liked (saved) songs", **RO})
async def spotify_get_liked_songs(params: LikedSongsInput) -> str:
    """Fetch the user's Liked Songs, most-recently-saved first, paginating as needed.

    Returns: {count, items:[{name, artists, album, id, uri, added_at}]}.
    """
    try:
        rows = await _paginate("/me/tracks", item_limit=params.limit)
        items = []
        for r in rows:
            t = _fmt_track(r.get("track", {}))
            t["added_at"] = r.get("added_at")
            items.append(t)
        return _ok({"count": len(items), "items": items})
    except Exception as e:
        return _err(e)


class SavedAlbumsInput(BaseModel):
    model_config = _CFG
    limit: int = Field(default=50, description="How many saved albums to fetch (1-500).", ge=1, le=500)


@mcp.tool(name="spotify_get_saved_albums", annotations={"title": "Get my saved albums", **RO})
async def spotify_get_saved_albums(params: SavedAlbumsInput) -> str:
    """Fetch albums saved in the user's library. Returns {count, items:[album]}."""
    try:
        rows = await _paginate("/me/albums", item_limit=params.limit)
        return _ok({"count": len(rows), "items": [_fmt_album(r.get("album", {})) for r in rows]})
    except Exception as e:
        return _err(e)


class TopItemsInput(BaseModel):
    model_config = _CFG
    item_type: TopType = Field(..., description="Whether to return your top 'artists' or top 'tracks'.")
    time_range: TimeRange = Field(default=TimeRange.medium, description="Window: short_term (~4wk), medium_term (~6mo), long_term (years).")
    limit: int = Field(default=20, description="How many to return (1-50).", ge=1, le=50)


@mcp.tool(name="spotify_get_top_items", annotations={"title": "Get my top artists or tracks", **RO})
async def spotify_get_top_items(params: TopItemsInput) -> str:
    """Return the user's most-listened artists or tracks over a time window.

    This is the best proxy for 'what I listen to a lot'. Returns {count, items}.
    """
    try:
        data = await _api("GET", f"/me/top/{params.item_type.value}",
                          params={"time_range": params.time_range.value, "limit": params.limit})
        fmt = _fmt_artist if params.item_type == TopType.artists else _fmt_track
        return _ok({"count": len(data.get("items", [])), "items": [fmt(x) for x in data.get("items", [])]})
    except Exception as e:
        return _err(e)


class RecentlyPlayedInput(BaseModel):
    model_config = _CFG
    limit: int = Field(default=25, description="How many recently played tracks (1-50).", ge=1, le=50)


@mcp.tool(name="spotify_get_recently_played", annotations={"title": "Get recently played tracks", **RO})
async def spotify_get_recently_played(params: RecentlyPlayedInput) -> str:
    """Return the user's recently played tracks, newest first. Returns {count, items:[{...,played_at}]}."""
    try:
        data = await _api("GET", "/me/player/recently-played", params={"limit": params.limit})
        items = []
        for r in data.get("items", []):
            t = _fmt_track(r.get("track", {}))
            t["played_at"] = r.get("played_at")
            items.append(t)
        return _ok({"count": len(items), "items": items})
    except Exception as e:
        return _err(e)


class FollowedArtistsInput(BaseModel):
    model_config = _CFG
    limit: int = Field(default=50, description="How many followed artists to fetch (1-500).", ge=1, le=500)


@mcp.tool(name="spotify_get_followed_artists", annotations={"title": "Get artists I follow", **RO})
async def spotify_get_followed_artists(params: FollowedArtistsInput) -> str:
    """Return artists the user follows. Returns {count, items:[artist]}."""
    try:
        # /me/following uses cursor pagination keyed under "artists".
        collected: List[dict] = []
        after: Optional[str] = None
        while len(collected) < params.limit:
            page = await _api("GET", "/me/following",
                              params={"type": "artist", "limit": min(50, params.limit - len(collected)),
                                      **({"after": after} if after else {})})
            block = page.get("artists", {})
            items = block.get("items", [])
            collected.extend(items)
            after = (block.get("cursors") or {}).get("after")
            if not after or not items:
                break
        return _ok({"count": len(collected), "items": [_fmt_artist(a) for a in collected]})
    except Exception as e:
        return _err(e)


# ----- Playlists ----------------------------------------------------------- #

class ListPlaylistsInput(BaseModel):
    model_config = _CFG
    limit: int = Field(default=50, description="How many playlists to fetch (1-500).", ge=1, le=500)


@mcp.tool(name="spotify_list_playlists", annotations={"title": "List my playlists", **RO})
async def spotify_list_playlists(params: ListPlaylistsInput) -> str:
    """List playlists owned by or followed by the user. Returns {count, items:[playlist]} including each playlist's id."""
    try:
        rows = await _paginate("/me/playlists", item_limit=params.limit)
        return _ok({"count": len(rows), "items": [_fmt_playlist(p) for p in rows if p]})
    except Exception as e:
        return _err(e)


class PlaylistIdInput(BaseModel):
    model_config = _CFG
    playlist_id: str = Field(..., description="Spotify playlist id (the part after /playlist/) or full spotify:playlist: URI.", min_length=1)


@mcp.tool(name="spotify_get_playlist", annotations={"title": "Get playlist details", **RO})
async def spotify_get_playlist(params: PlaylistIdInput) -> str:
    """Return a single playlist's metadata (name, owner, track count, description)."""
    try:
        pid = params.playlist_id.split(":")[-1]
        data = await _api("GET", f"/playlists/{pid}",
                          params={"fields": "id,uri,name,description,public,owner(display_name),tracks(total)"})
        return _ok(_fmt_playlist(data))
    except Exception as e:
        return _err(e)


class PlaylistTracksInput(BaseModel):
    model_config = _CFG
    playlist_id: str = Field(..., description="Spotify playlist id or full URI.", min_length=1)
    limit: int = Field(default=100, description="How many tracks to fetch (1-1000).", ge=1, le=1000)
    market: Optional[str] = Field(default=None, description="ISO country code, e.g. 'CA'.", max_length=2)


@mcp.tool(name="spotify_get_playlist_tracks", annotations={"title": "Get the tracks inside a playlist", **RO})
async def spotify_get_playlist_tracks(params: PlaylistTracksInput) -> str:
    """Enumerate the actual tracks inside a playlist, in order.

    This is the capability the default Spotify connector lacks. Returns
    {count, items:[{name, artists, album, id, uri, added_at}]}.
    """
    try:
        pid = params.playlist_id.split(":")[-1]
        # A missing market + additional_types=track is a known trigger for 403s
        # on /playlists/{id}/tracks even when scopes are fine. Default the market
        # to the account's country and request explicit fields.
        market = params.market
        if not market:
            try:
                market = (await _api("GET", "/me")).get("country")
            except Exception:
                market = None
        p = {"limit": 50,
             "fields": "items(added_at,track(name,id,uri,duration_ms,popularity,"
                       "artists(name),album(name))),next"}
        if market:
            p["market"] = market
        rows = await _paginate(f"/playlists/{pid}/tracks", params=p, item_limit=params.limit)
        items = []
        for r in rows:
            t = _fmt_track(r.get("track", {}))
            if not t:
                continue
            t["added_at"] = r.get("added_at")
            items.append(t)
        return _ok({"count": len(items), "items": items})
    except Exception as e:
        return _err(e)


class CreatePlaylistInput(BaseModel):
    model_config = _CFG
    name: str = Field(..., description="Playlist name.", min_length=1, max_length=200)
    description: Optional[str] = Field(default=None, description="Optional playlist description.", max_length=300)
    public: bool = Field(default=False, description="Whether the playlist is public. Defaults to private.")
    track_uris: Optional[List[str]] = Field(default=None, description="Optional track ids or spotify:track: URIs to add on creation.", max_length=100)


@mcp.tool(name="spotify_create_playlist", annotations={"title": "Create a playlist", **WRITE})
async def spotify_create_playlist(params: CreatePlaylistInput) -> str:
    """Create a new playlist for the user, optionally seeding it with tracks.

    Returns the new playlist {id, uri, name, ...}. Use spotify_add_tracks_to_playlist
    to add more than 100 tracks afterward.
    """
    try:
        me = await _api("GET", "/me")
        body = {"name": params.name, "public": params.public}
        if params.description:
            body["description"] = params.description
        pl = await _api("POST", f"/users/{me['id']}/playlists", json_body=body)
        if params.track_uris:
            await _api("POST", f"/playlists/{pl['id']}/tracks",
                       json_body={"uris": _to_uris(params.track_uris)})
            pl = await _api("GET", f"/playlists/{pl['id']}",
                            params={"fields": "id,uri,name,description,public,owner(display_name),tracks(total)"})
        return _ok(_fmt_playlist(pl))
    except Exception as e:
        return _err(e)


class AddTracksInput(BaseModel):
    model_config = _CFG
    playlist_id: str = Field(..., description="Target playlist id or URI.", min_length=1)
    track_uris: List[str] = Field(..., description="Track ids or spotify:track: URIs to add.", min_length=1, max_length=100)
    position: Optional[int] = Field(default=None, description="Zero-based insert position. Omit to append.", ge=0)


@mcp.tool(name="spotify_add_tracks_to_playlist", annotations={"title": "Add tracks to a playlist", **WRITE})
async def spotify_add_tracks_to_playlist(params: AddTracksInput) -> str:
    """Add up to 100 tracks to a playlist (append, or insert at a position). Returns the new snapshot_id."""
    try:
        pid = params.playlist_id.split(":")[-1]
        body: Dict[str, Any] = {"uris": _to_uris(params.track_uris)}
        if params.position is not None:
            body["position"] = params.position
        data = await _api("POST", f"/playlists/{pid}/tracks", json_body=body)
        return _ok({"added": len(params.track_uris), "snapshot_id": data.get("snapshot_id")})
    except Exception as e:
        return _err(e)


class RemoveTracksInput(BaseModel):
    model_config = _CFG
    playlist_id: str = Field(..., description="Target playlist id or URI.", min_length=1)
    track_uris: List[str] = Field(..., description="Track ids or spotify:track: URIs to remove (all occurrences).", min_length=1, max_length=100)


@mcp.tool(name="spotify_remove_tracks_from_playlist", annotations={"title": "Remove tracks from a playlist", **DESTRUCTIVE})
async def spotify_remove_tracks_from_playlist(params: RemoveTracksInput) -> str:
    """Remove all occurrences of the given tracks from a playlist. Returns the new snapshot_id."""
    try:
        pid = params.playlist_id.split(":")[-1]
        tracks = [{"uri": u} for u in _to_uris(params.track_uris)]
        data = await _api("DELETE", f"/playlists/{pid}/tracks", json_body={"tracks": tracks})
        return _ok({"removed": len(tracks), "snapshot_id": data.get("snapshot_id")})
    except Exception as e:
        return _err(e)


class ModifyPlaylistInput(BaseModel):
    model_config = _CFG
    playlist_id: str = Field(..., description="Target playlist id or URI.", min_length=1)
    name: Optional[str] = Field(default=None, description="New name.", max_length=200)
    description: Optional[str] = Field(default=None, description="New description.", max_length=300)
    public: Optional[bool] = Field(default=None, description="New public/private setting.")


@mcp.tool(name="spotify_change_playlist_details", annotations={"title": "Edit playlist name/description/visibility", **WRITE})
async def spotify_change_playlist_details(params: ModifyPlaylistInput) -> str:
    """Update a playlist's name, description, and/or public flag."""
    try:
        pid = params.playlist_id.split(":")[-1]
        body = {k: v for k, v in {"name": params.name, "description": params.description,
                                  "public": params.public}.items() if v is not None}
        if not body:
            return _err(SpotifyError("Provide at least one of name, description, or public."))
        await _api("PUT", f"/playlists/{pid}", json_body=body)
        return _ok({"updated": True, "fields": list(body.keys())})
    except Exception as e:
        return _err(e)


# ----- Catalog objects ----------------------------------------------------- #

class TrackIdsInput(BaseModel):
    model_config = _CFG
    track_ids: List[str] = Field(..., description="One or more track ids or spotify:track: URIs.", min_length=1, max_length=50)


@mcp.tool(name="spotify_get_tracks", annotations={"title": "Get details for tracks", **RO})
async def spotify_get_tracks(params: TrackIdsInput) -> str:
    """Return full details for up to 50 tracks. Returns {count, items:[track]}."""
    try:
        ids = ",".join(x.split(":")[-1] for x in params.track_ids)
        data = await _api("GET", "/tracks", params={"ids": ids})
        return _ok({"count": len(data.get("tracks", [])), "items": [_fmt_track(t) for t in data.get("tracks", []) if t]})
    except Exception as e:
        return _err(e)


class ArtistIdsInput(BaseModel):
    model_config = _CFG
    artist_ids: List[str] = Field(..., description="One or more artist ids or spotify:artist: URIs.", min_length=1, max_length=50)


@mcp.tool(name="spotify_get_artists", annotations={"title": "Get details for artists (incl. genres)", **RO})
async def spotify_get_artists(params: ArtistIdsInput) -> str:
    """Return details for up to 50 artists, including genres (useful for similarity). Returns {count, items:[artist]}."""
    try:
        ids = ",".join(x.split(":")[-1] for x in params.artist_ids)
        data = await _api("GET", "/artists", params={"ids": ids})
        return _ok({"count": len(data.get("artists", [])), "items": [_fmt_artist(a) for a in data.get("artists", []) if a]})
    except Exception as e:
        return _err(e)


class ArtistAlbumsInput(BaseModel):
    model_config = _CFG
    artist_id: str = Field(..., description="Artist id or spotify:artist: URI.", min_length=1)
    limit: int = Field(default=20, description="How many albums to fetch (1-50).", ge=1, le=50)
    include_groups: Optional[str] = Field(default="album,single", description="Comma list of: album, single, appears_on, compilation.")


@mcp.tool(name="spotify_get_artist_albums", annotations={"title": "Get an artist's albums", **RO})
async def spotify_get_artist_albums(params: ArtistAlbumsInput) -> str:
    """Return albums/singles for an artist. Returns {count, items:[album]}."""
    try:
        aid = params.artist_id.split(":")[-1]
        p = {"limit": params.limit}
        if params.include_groups:
            p["include_groups"] = params.include_groups
        rows = await _paginate(f"/artists/{aid}/albums", params=p, item_limit=params.limit)
        return _ok({"count": len(rows), "items": [_fmt_album(a) for a in rows]})
    except Exception as e:
        return _err(e)


class AlbumTracksInput(BaseModel):
    model_config = _CFG
    album_id: str = Field(..., description="Album id or spotify:album: URI.", min_length=1)
    limit: int = Field(default=50, description="How many tracks to fetch (1-50).", ge=1, le=50)


@mcp.tool(name="spotify_get_album_tracks", annotations={"title": "Get the tracks on an album", **RO})
async def spotify_get_album_tracks(params: AlbumTracksInput) -> str:
    """Return the track list of an album. Returns {count, items:[track]}."""
    try:
        alid = params.album_id.split(":")[-1]
        rows = await _paginate(f"/albums/{alid}/tracks", item_limit=params.limit)
        return _ok({"count": len(rows), "items": [_fmt_track(t) for t in rows]})
    except Exception as e:
        return _err(e)


# ----- Library / follow writes -------------------------------------------- #

@mcp.tool(name="spotify_save_tracks", annotations={"title": "Like (save) tracks", **WRITE})
async def spotify_save_tracks(params: TrackIdsInput) -> str:
    """Add up to 50 tracks to the user's Liked Songs."""
    try:
        ids = [x.split(":")[-1] for x in params.track_ids]
        await _api("PUT", "/me/tracks", json_body={"ids": ids})
        return _ok({"saved": len(ids)})
    except Exception as e:
        return _err(e)


@mcp.tool(name="spotify_remove_saved_tracks", annotations={"title": "Unlike (remove) saved tracks", **DESTRUCTIVE})
async def spotify_remove_saved_tracks(params: TrackIdsInput) -> str:
    """Remove up to 50 tracks from the user's Liked Songs."""
    try:
        ids = [x.split(":")[-1] for x in params.track_ids]
        await _api("DELETE", "/me/tracks", json_body={"ids": ids})
        return _ok({"removed": len(ids)})
    except Exception as e:
        return _err(e)


@mcp.tool(name="spotify_follow_artists", annotations={"title": "Follow artists", **WRITE})
async def spotify_follow_artists(params: ArtistIdsInput) -> str:
    """Follow up to 50 artists."""
    try:
        ids = ",".join(x.split(":")[-1] for x in params.artist_ids)
        await _api("PUT", "/me/following", params={"type": "artist", "ids": ids})
        return _ok({"followed": len(params.artist_ids)})
    except Exception as e:
        return _err(e)


@mcp.tool(name="spotify_unfollow_artists", annotations={"title": "Unfollow artists", **DESTRUCTIVE})
async def spotify_unfollow_artists(params: ArtistIdsInput) -> str:
    """Unfollow up to 50 artists."""
    try:
        ids = ",".join(x.split(":")[-1] for x in params.artist_ids)
        await _api("DELETE", "/me/following", params={"type": "artist", "ids": ids})
        return _ok({"unfollowed": len(params.artist_ids)})
    except Exception as e:
        return _err(e)


# ----- Playback ------------------------------------------------------------ #

@mcp.tool(name="spotify_get_currently_playing", annotations={"title": "Get what's currently playing", **RO})
async def spotify_get_currently_playing(params: Empty) -> str:
    """Return the track currently playing on the user's account, if any. Returns {is_playing, track} or {is_playing: false}."""
    try:
        data = await _api("GET", "/me/player/currently-playing")
        if not data or not data.get("item"):
            return _ok({"is_playing": False})
        return _ok({"is_playing": data.get("is_playing"), "track": _fmt_track(data["item"])})
    except Exception as e:
        return _err(e)


# ----- Similarity (replacement for deprecated /recommendations) ------------ #

class FindSimilarInput(BaseModel):
    model_config = _CFG
    seed_type: SearchType = Field(default=SearchType.track, description="Use a 'track' or 'artist' as the similarity seed.")
    seed: str = Field(..., description="Seed name (e.g. 'Travis Scott') OR a track/artist id or URI.", min_length=1)
    limit: int = Field(default=20, description="How many similar tracks to return (1-50).", ge=1, le=50)
    market: Optional[str] = Field(default=None, description="ISO country code, e.g. 'CA'.", max_length=2)


async def _resolve_seed_name(seed: str, seed_type: SearchType) -> tuple[str, Optional[str]]:
    """Return (display_name, artist_name) for a seed given as a name, id, or URI."""
    looks_like_id = ":" in seed or (len(seed) == 22 and seed.isalnum())
    if not looks_like_id:
        return seed, (seed if seed_type == SearchType.artist else None)
    sid = seed.split(":")[-1]
    if seed_type == SearchType.track:
        t = await _api("GET", f"/tracks/{sid}")
        return t.get("name", seed), _artists_str(t)
    a = await _api("GET", f"/artists/{sid}")
    return a.get("name", seed), a.get("name")


async def _lastfm_similar(name: str, artist: Optional[str], seed_type: SearchType, limit: int) -> List[str]:
    """Query Last.fm for similar tracks/artists; return a list of 'artist - title' (or artist) strings."""
    method = "track.getsimilar" if seed_type == SearchType.track else "artist.getsimilar"
    q = {"method": method, "api_key": LASTFM_API_KEY, "format": "json", "limit": limit * 2}
    if seed_type == SearchType.track:
        q["track"] = name
        q["artist"] = artist or ""
    else:
        q["artist"] = name
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.get(LASTFM_API_BASE, params=q)
    if resp.status_code != 200:
        return []
    data = resp.json()
    out: List[str] = []
    if seed_type == SearchType.track:
        for m in (data.get("similartracks", {}) or {}).get("track", []):
            out.append(f"{(m.get('artist') or {}).get('name','')} {m.get('name','')}".strip())
    else:
        for a in (data.get("similarartists", {}) or {}).get("artist", []):
            out.append(a.get("name", ""))
    return [s for s in out if s]


@mcp.tool(name="spotify_find_similar", annotations={"title": "Find songs similar to a seed track/artist", **RO})
async def spotify_find_similar(params: FindSimilarInput) -> str:
    """Find tracks similar to a seed track or artist and resolve them to Spotify.

    Spotify removed its native Recommendations/Related-Artists/Audio-Features
    endpoints for new apps, so this tool reconstructs similarity:
      - If LASTFM_API_KEY is set, it uses Last.fm's similar-track / similar-artist
        graph, then resolves each suggestion back to a Spotify track via search.
      - Otherwise it falls back to the seed artist's genres and pulls tracks via
        catalog genre search.
    Returns: {method, seed, count, items:[track]}.
    """
    try:
        name, artist = await _resolve_seed_name(params.seed, params.seed_type)
        results: List[dict] = []
        seen: set[str] = set()
        method_used = ""

        if LASTFM_API_KEY:
            method_used = "lastfm"
            suggestions = await _lastfm_similar(name, artist, params.seed_type, params.limit)
            for s in suggestions:
                if len(results) >= params.limit:
                    break
                p = {"q": s, "type": "track", "limit": 1}
                if params.market:
                    p["market"] = params.market
                sr = await _api("GET", "/search", params=p)
                hits = (sr.get("tracks") or {}).get("items", [])
                if hits and hits[0]["id"] not in seen:
                    seen.add(hits[0]["id"])
                    results.append(_fmt_track(hits[0]))

        if not results:
            # Fallback: genre + catalog search around the seed artist.
            method_used = method_used or "genre_fallback"
            genres: List[str] = []
            if params.seed_type == SearchType.artist:
                aid = params.seed.split(":")[-1] if (":" in params.seed or len(params.seed) == 22) else None
                if not aid:
                    sr = await _api("GET", "/search", params={"q": name, "type": "artist", "limit": 1})
                    a_items = (sr.get("artists") or {}).get("items", [])
                    aid = a_items[0]["id"] if a_items else None
                if aid:
                    a = await _api("GET", f"/artists/{aid}")
                    genres = a.get("genres", [])
            queries = [f'genre:"{g}"' for g in genres[:3]] or [name]
            for q in queries:
                if len(results) >= params.limit:
                    break
                p = {"q": q, "type": "track", "limit": max(5, params.limit)}
                if params.market:
                    p["market"] = params.market
                sr = await _api("GET", "/search", params=p)
                for t in (sr.get("tracks") or {}).get("items", []):
                    if len(results) >= params.limit:
                        break
                    if t["id"] not in seen:
                        seen.add(t["id"])
                        results.append(_fmt_track(t))

        return _ok({"method": method_used, "seed": name, "count": len(results), "items": results})
    except Exception as e:
        return _err(e)


async def debug_playlist(request: Request) -> JSONResponse:
    """TEMPORARY diagnostic: probes several Spotify read endpoints and returns
    raw statuses to isolate whether the 403 is app-wide or playlist-specific.
    Visit /debug/playlist/<id>. Remove later.
    """
    pid = request.path_params["pid"]
    out: Dict[str, Any] = {"playlist_id": pid}
    try:
        token = await _ensure_token()
        hdr = {"Authorization": f"Bearer {token}"}
        async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
            me = await client.get(f"{SPOTIFY_API_BASE}/me", headers=hdr)
            mj = me.json() if me.status_code == 200 else {}
            out["me"] = {"id": mj.get("id"), "country": mj.get("country"),
                         "product": mj.get("product")}

            # 1) target playlist metadata (known-good)
            meta = await client.get(f"{SPOTIFY_API_BASE}/playlists/{pid}", headers=hdr,
                                    params={"fields": "id,name,owner(id)"})
            out["1_target_meta_status"] = meta.status_code

            # 2) target playlist TRACKS (the failing call)
            tr = await client.get(f"{SPOTIFY_API_BASE}/playlists/{pid}/tracks",
                                  headers=hdr, params={"limit": 3})
            out["2_target_tracks_status"] = tr.status_code

            # 3) Liked Songs (different endpoint, needs user-library-read)
            lk = await client.get(f"{SPOTIFY_API_BASE}/me/tracks", headers=hdr,
                                  params={"limit": 3})
            out["3_liked_songs_status"] = lk.status_code

            # 4) Top tracks (needs user-top-read)
            tp = await client.get(f"{SPOTIFY_API_BASE}/me/top/tracks", headers=hdr,
                                  params={"limit": 3})
            out["4_top_tracks_status"] = tp.status_code

            # 5) tracks of the FIRST playlist returned by /me/playlists
            pls = await client.get(f"{SPOTIFY_API_BASE}/me/playlists", headers=hdr,
                                   params={"limit": 5})
            out["5_list_playlists_status"] = pls.status_code
            if pls.status_code == 200:
                items = pls.json().get("items", [])
                probed = []
                for p in items[:3]:
                    if not p:
                        continue
                    r = await client.get(f"{SPOTIFY_API_BASE}/playlists/{p['id']}/tracks",
                                         headers=hdr, params={"limit": 1})
                    probed.append({"name": p.get("name"),
                                   "owner": (p.get("owner") or {}).get("id"),
                                   "tracks_status": r.status_code})
                out["6_other_playlists_tracks"] = probed
    except Exception as e:
        out["exception"] = f"{type(e).__name__}: {e}"
    return JSONResponse(out)


# --------------------------------------------------------------------------- #
# OAuth 2.0 layer for the Claude connector
#
# Claude's custom-connector flow speaks MCP's OAuth profile: it discovers
# metadata, dynamically registers a client, runs an authorization-code + PKCE
# flow, and then calls /mcp with a Bearer token. This server is single-user, so
# the authorization step auto-approves (the security boundary is your private
# server URL). Tokens are stateless HMAC-signed blobs keyed off CLIENT_SECRET,
# so they survive free-tier restarts without server-side storage.
# --------------------------------------------------------------------------- #

_SIGNING_KEY = hashlib.sha256(("mcp-oauth-v1:" + (CLIENT_SECRET or "dev")).encode()).digest()


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _b64u_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _sign(payload: dict) -> str:
    raw = json.dumps(payload, separators=(",", ":")).encode()
    p = _b64u(raw)
    sig = _b64u(hmac.new(_SIGNING_KEY, p.encode(), hashlib.sha256).digest())
    return f"{p}.{sig}"


def _unsign(token: str) -> Optional[dict]:
    try:
        p, sig = token.split(".", 1)
        expected = _b64u(hmac.new(_SIGNING_KEY, p.encode(), hashlib.sha256).digest())
        if not hmac.compare_digest(expected, sig):
            return None
        payload = json.loads(_b64u_decode(p))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


def _verify_access(token: str) -> bool:
    p = _unsign(token)
    return bool(p and p.get("typ") == "access")


def _issue_tokens() -> dict:
    now = int(time.time())
    return {
        "access_token": _sign({"typ": "access", "exp": now + 3600}),
        "token_type": "Bearer",
        "expires_in": 3600,
        "refresh_token": _sign({"typ": "refresh", "exp": now + 30 * 86400}),
        "scope": "mcp",
    }


def _base_url(request: Request) -> str:
    proto = request.headers.get("x-forwarded-proto", "https")
    host = request.headers.get("host", "")
    return f"{proto}://{host}"


async def _read_params(request: Request) -> dict:
    ctype = request.headers.get("content-type", "")
    if "application/json" in ctype:
        try:
            return await request.json()
        except Exception:
            return {}
    raw = (await request.body()).decode()
    return {k: v[0] for k, v in urllib.parse.parse_qs(raw).items()}


async def oauth_protected_resource(request: Request) -> JSONResponse:
    base = _base_url(request)
    return JSONResponse({
        "resource": f"{base}/mcp",
        "authorization_servers": [base],
        "scopes_supported": ["mcp"],
        "bearer_methods_supported": ["header"],
    })


async def oauth_authorization_server(request: Request) -> JSONResponse:
    base = _base_url(request)
    return JSONResponse({
        "issuer": base,
        "authorization_endpoint": f"{base}/oauth/authorize",
        "token_endpoint": f"{base}/oauth/token",
        "registration_endpoint": f"{base}/oauth/register",
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code", "refresh_token"],
        "code_challenge_methods_supported": ["S256"],
        "token_endpoint_auth_methods_supported": ["none"],
        "scopes_supported": ["mcp"],
    })


async def oauth_register(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    resp = {
        "client_id": "mcp-" + secrets.token_urlsafe(16),
        "client_id_issued_at": int(time.time()),
        "redirect_uris": body.get("redirect_uris", []),
        "token_endpoint_auth_method": "none",
        "grant_types": ["authorization_code", "refresh_token"],
        "response_types": ["code"],
    }
    for k in ("client_name", "scope"):
        if k in body:
            resp[k] = body[k]
    return JSONResponse(resp, status_code=201)


async def oauth_authorize(request: Request) -> JSONResponse | RedirectResponse:
    q = request.query_params
    redirect_uri = q.get("redirect_uri")
    code_challenge = q.get("code_challenge")
    if not redirect_uri or not code_challenge or q.get("code_challenge_method") != "S256":
        return JSONResponse(
            {"error": "invalid_request",
             "error_description": "redirect_uri and S256 PKCE are required"},
            status_code=400,
        )
    code = _sign({"typ": "code", "cc": code_challenge, "ru": redirect_uri,
                  "exp": int(time.time()) + 300})
    sep = "&" if "?" in redirect_uri else "?"
    location = f"{redirect_uri}{sep}code={urllib.parse.quote(code)}"
    state = q.get("state")
    if state:
        location += f"&state={urllib.parse.quote(state)}"
    return RedirectResponse(location, status_code=302)


async def oauth_token(request: Request) -> JSONResponse:
    params = await _read_params(request)
    grant = params.get("grant_type")
    if grant == "authorization_code":
        payload = _unsign(params.get("code", ""))
        if not payload or payload.get("typ") != "code":
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        if payload.get("ru") != params.get("redirect_uri"):
            return JSONResponse({"error": "invalid_grant",
                                 "error_description": "redirect_uri mismatch"}, status_code=400)
        check = _b64u(hashlib.sha256(params.get("code_verifier", "").encode()).digest())
        if not hmac.compare_digest(check, payload.get("cc", "")):
            return JSONResponse({"error": "invalid_grant",
                                 "error_description": "PKCE verification failed"}, status_code=400)
        return JSONResponse(_issue_tokens())
    if grant == "refresh_token":
        payload = _unsign(params.get("refresh_token", ""))
        if not payload or payload.get("typ") != "refresh":
            return JSONResponse({"error": "invalid_grant"}, status_code=400)
        return JSONResponse(_issue_tokens())
    return JSONResponse({"error": "unsupported_grant_type"}, status_code=400)


class MCPAuthMiddleware:
    """Pure-ASGI gate on /mcp. Passes authorized requests through untouched
    (so MCP's streaming responses are not buffered); returns a 401 with a
    WWW-Authenticate pointer otherwise, which kicks off Claude's OAuth flow.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") == "http":
            path = scope.get("path", "")
            if path == "/mcp" or path.startswith("/mcp/"):
                headers = dict(scope.get("headers") or [])
                auth = headers.get(b"authorization", b"").decode()
                token = auth[7:] if auth[:7].lower() == "bearer " else ""
                if not _verify_access(token):
                    proto = headers.get(b"x-forwarded-proto", b"https").decode()
                    host = headers.get(b"host", b"").decode()
                    meta = f"{proto}://{host}/.well-known/oauth-protected-resource"
                    await send({
                        "type": "http.response.start",
                        "status": 401,
                        "headers": [
                            (b"content-type", b"application/json"),
                            (b"www-authenticate", f'Bearer resource_metadata="{meta}"'.encode()),
                        ],
                    })
                    await send({"type": "http.response.body",
                                "body": b'{"error":"invalid_token"}'})
                    return
        await self.app(scope, receive, send)


# --------------------------------------------------------------------------- #
# Spotify OAuth bootstrap + health routes
# --------------------------------------------------------------------------- #

async def health(_request: Request) -> JSONResponse:
    authorized = bool(_token_state.get("refresh_token"))
    return JSONResponse({
        "status": "ok",
        "service": "spotify_mcp",
        "authorized": authorized,
        "client_configured": bool(CLIENT_ID and CLIENT_SECRET),
        "redirect_uri_set": bool(REDIRECT_URI),
        "lastfm_enabled": bool(LASTFM_API_KEY),
        "connector_oauth": "enabled",
        "mcp_endpoint": "/mcp",
    })


async def login(_request: Request) -> RedirectResponse | HTMLResponse:
    if not (CLIENT_ID and REDIRECT_URI):
        return HTMLResponse("<h3>Server not configured.</h3><p>Set SPOTIFY_CLIENT_ID, "
                            "SPOTIFY_CLIENT_SECRET, and SPOTIFY_REDIRECT_URI, then redeploy.</p>",
                            status_code=500)
    state = secrets.token_urlsafe(16)
    _oauth_states.add(state)
    qs = urllib.parse.urlencode({
        "client_id": CLIENT_ID, "response_type": "code", "redirect_uri": REDIRECT_URI,
        "scope": SCOPES, "state": state, "show_dialog": "true",
    })
    return RedirectResponse(f"{SPOTIFY_ACCOUNTS}/authorize?{qs}")


async def callback(request: Request) -> HTMLResponse:
    err = request.query_params.get("error")
    if err:
        return HTMLResponse(f"<h3>Authorization failed:</h3><pre>{err}</pre>", status_code=400)
    code = request.query_params.get("code")
    state = request.query_params.get("state")
    if not code or state not in _oauth_states:
        return HTMLResponse("<h3>Invalid or expired request.</h3><p>Start again at /login.</p>", status_code=400)
    _oauth_states.discard(state)
    async with httpx.AsyncClient(timeout=HTTP_TIMEOUT) as client:
        resp = await client.post(
            f"{SPOTIFY_ACCOUNTS}/api/token",
            headers={"Authorization": _basic_auth_header(),
                     "Content-Type": "application/x-www-form-urlencoded"},
            data={"grant_type": "authorization_code", "code": code, "redirect_uri": REDIRECT_URI},
        )
    if resp.status_code != 200:
        return HTMLResponse(f"<h3>Token exchange failed ({resp.status_code}).</h3><pre>{resp.text}</pre>",
                            status_code=400)
    payload = resp.json()
    refresh_token = payload.get("refresh_token", "")
    _token_state["access_token"] = payload.get("access_token")
    _token_state["refresh_token"] = refresh_token or _token_state.get("refresh_token")
    _token_state["expires_at"] = time.time() + int(payload.get("expires_in", 3600))
    return HTMLResponse(f"""
        <h2>✅ Authorized</h2>
        <p>Spotify is connected for this running instance. To make it survive
        restarts, set this as the <code>SPOTIFY_REFRESH_TOKEN</code> environment
        variable on your host, then redeploy:</p>
        <pre style="white-space:pre-wrap;word-break:break-all;background:#f4f4f4;padding:12px;border-radius:8px;">{refresh_token}</pre>
        <p>Then add <code>{REDIRECT_URI.rsplit('/',1)[0]}/mcp</code> to Claude as a custom connector.</p>
        <p style="color:#a00;">Treat this token like a password — it grants control of your Spotify account.</p>
    """)


# --------------------------------------------------------------------------- #
# Build the ASGI app: MCP at /mcp (OAuth-gated) + OAuth + helper routes
# --------------------------------------------------------------------------- #

_starlette = mcp.streamable_http_app()  # Starlette app; mounts the MCP endpoint at /mcp

# OAuth discovery metadata (served at root and path-suffixed variants clients probe)
for _p in ("/.well-known/oauth-protected-resource", "/.well-known/oauth-protected-resource/mcp"):
    _starlette.add_route(_p, oauth_protected_resource, methods=["GET"])
for _p in ("/.well-known/oauth-authorization-server", "/.well-known/oauth-authorization-server/mcp",
           "/.well-known/openid-configuration"):
    _starlette.add_route(_p, oauth_authorization_server, methods=["GET"])

# OAuth endpoints
_starlette.add_route("/oauth/register", oauth_register, methods=["POST"])
_starlette.add_route("/oauth/authorize", oauth_authorize, methods=["GET"])
_starlette.add_route("/oauth/token", oauth_token, methods=["POST"])

# Health + Spotify bootstrap
_starlette.add_route("/health", health, methods=["GET"])
_starlette.add_route("/", health, methods=["GET"])
_starlette.add_route("/login", login, methods=["GET"])
_starlette.add_route("/callback", callback, methods=["GET"])
_starlette.add_route("/debug/playlist/{pid}", debug_playlist, methods=["GET"])  # TEMP diagnostic

# Gate /mcp behind the connector OAuth token (everything else stays open).
app = MCPAuthMiddleware(_starlette)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
