from __future__ import annotations

import csv
import io
import os
import secrets
import threading
import time
from collections import defaultdict, deque

from flask import Flask, Response, flash, redirect, render_template, request, session, url_for

from buffalo_auth_core import AuthConfig, AuthEngine


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024
app.secret_key = os.environ.get("SECUREFACE_SECRET_KEY") or secrets.token_hex(32)

engine = AuthEngine(AuthConfig())
RATE_LIMIT_LOCK = threading.Lock()
RATE_LIMIT_BUCKETS: dict[str, deque[float]] = defaultdict(deque)
RATE_LIMITS = {
    "register": {"max_attempts": 5, "window_seconds": 10 * 60},
    "authenticate": {"max_attempts": 10, "window_seconds": 10 * 60},
}
CHALLENGE_TTL_SECONDS = 3 * 60
LOCKOUT_THRESHOLD = 5
LOCKOUT_SECONDS = 10 * 60
LOCKOUTS: dict[str, float] = {}


def _warm_engine_models() -> None:
    try:
        engine._ensure_models()
        app.logger.info("Auth models warmed successfully.")
    except Exception as exc:
        app.logger.warning("Auth model warm-up skipped: %s", exc)


threading.Thread(target=_warm_engine_models, daemon=True).start()


def _client_address() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "local"


def _ensure_csrf_token() -> str:
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(24)
        session["csrf_token"] = token
    return token


def _challenge_instruction(challenge: dict[str, int | float | str]) -> str:
    action = str(challenge.get("action", "blink"))
    if action == "turn_left":
        return "Turn your face slightly to the left during the live capture."
    if action == "turn_right":
        return "Turn your face slightly to the right during the live capture."
    count = int(challenge.get("required_blinks", 1) or 1)
    return f"Blink {count} time{'s' if count != 1 else ''} during the live capture."


def _issue_challenge(flow_name: str) -> dict[str, int | float | str]:
    challenges = dict(session.get("capture_challenges", {}))
    challenge_type = secrets.choice(("blink", "turn_left", "turn_right"))
    challenge: dict[str, int | float | str] = {
        "nonce": secrets.token_urlsafe(18),
        "action": challenge_type,
        "issued_at": time.time(),
    }
    if challenge_type == "blink":
        challenge["required_blinks"] = 1
    challenges[flow_name] = challenge
    session["capture_challenges"] = challenges
    session.modified = True
    return challenge


def _render_flow(template_name: str, flow_name: str):
    challenge = _issue_challenge(flow_name)
    return render_template(
        template_name,
        csrf_token=_ensure_csrf_token(),
        challenge=challenge,
        challenge_instruction=_challenge_instruction(challenge),
    )


def _enforce_rate_limit(flow_name: str) -> None:
    config = RATE_LIMITS[flow_name]
    now = time.time()
    bucket_key = f"{flow_name}:{_client_address()}"
    with RATE_LIMIT_LOCK:
        bucket = RATE_LIMIT_BUCKETS[bucket_key]
        while bucket and (now - bucket[0]) > config["window_seconds"]:
            bucket.popleft()
        if len(bucket) >= config["max_attempts"]:
            raise RuntimeError("Too many attempts. Please wait a few minutes before trying again.")
        bucket.append(now)


def _lockout_key(flow_name: str, username: str | None = None) -> str:
    suffix = username.strip().lower() if username else _client_address()
    return f"{flow_name}:{suffix}"


def _ensure_not_locked(flow_name: str, username: str | None = None) -> None:
    key = _lockout_key(flow_name, username)
    now = time.time()
    until = LOCKOUTS.get(key)
    if until and until > now:
        wait_minutes = max(1, int((until - now) // 60) + 1)
        raise RuntimeError(f"Too many failed attempts. Try again in about {wait_minutes} minute(s).")
    if until and until <= now:
        LOCKOUTS.pop(key, None)


def _record_auth_outcome(flow_name: str, success: bool, username: str | None = None) -> None:
    key = _lockout_key(flow_name, username)
    rate_key = f"fail:{key}"
    now = time.time()
    with RATE_LIMIT_LOCK:
        bucket = RATE_LIMIT_BUCKETS[rate_key]
        while bucket and (now - bucket[0]) > LOCKOUT_SECONDS:
            bucket.popleft()
        if success:
            bucket.clear()
            LOCKOUTS.pop(key, None)
            return
        bucket.append(now)
        if len(bucket) >= LOCKOUT_THRESHOLD:
            LOCKOUTS[key] = now + LOCKOUT_SECONDS


def _verify_request_security(flow_name: str) -> dict[str, int | float | str]:
    _enforce_rate_limit(flow_name)

    submitted_csrf = (request.form.get("csrf_token", "") or "").strip()
    expected_csrf = session.get("csrf_token", "")
    if not expected_csrf or not submitted_csrf or not secrets.compare_digest(submitted_csrf, expected_csrf):
        raise RuntimeError("Your session expired. Refresh the page and try again.")

    submitted_nonce = (request.form.get("challenge_nonce", "") or "").strip()
    challenges = dict(session.get("capture_challenges", {}))
    challenge = challenges.get(flow_name)
    if not challenge or not submitted_nonce or not secrets.compare_digest(submitted_nonce, challenge.get("nonce", "")):
        raise RuntimeError("Capture challenge is invalid. Refresh the page and try again.")

    issued_at = float(challenge.get("issued_at", 0.0))
    if (time.time() - issued_at) > CHALLENGE_TTL_SECONDS:
        raise RuntimeError("Capture challenge expired. Refresh the page and try again.")

    challenges.pop(flow_name, None)
    session["capture_challenges"] = challenges
    session.modified = True
    return challenge


def _get_submission_inputs():
    webcam_frames = [frame for frame in request.files.getlist("webcam_frames") if frame and frame.filename]
    video = request.files.get("video")
    if video is not None and bool(video.filename):
        raise RuntimeError("Video upload is disabled. Use the live webcam capture.")
    return webcam_frames


def _csv_download_response(filename: str, rows: list[dict[str, object]], fieldnames: list[str]) -> Response:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field, "") for field in fieldnames})
    csv_body = buffer.getvalue()
    return Response(
        csv_body,
        mimetype="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@app.after_request
def _set_security_headers(response):
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "same-origin"
    response.headers["Cache-Control"] = "no-store"
    return response


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/audit")
def audit():
    events = engine.list_audit_events(limit=150)
    return render_template("audit.html", events=events)


@app.route("/audit/download")
def audit_download():
    events = engine.list_audit_events(limit=1000)
    return _csv_download_response(
        "secureface_audit_log.csv",
        events,
        ["id", "created_at", "event_type", "outcome", "username", "ip_address", "detail"],
    )


@app.route("/unknown-attempts")
def unknown_attempts():
    attempts = engine.list_unknown_attempts(limit=150)
    return render_template("unknown_attempts.html", attempts=attempts)


@app.route("/unknown-attempts/download")
def unknown_attempts_download():
    attempts = engine.list_unknown_attempts(limit=1000)
    return _csv_download_response(
        "secureface_unknown_alerts.csv",
        attempts,
        [
            "id",
            "created_at",
            "input_kind",
            "input_path",
            "run_dir",
            "preview_crop_path",
            "live_ratio",
            "mean_spoof",
            "session_score",
            "match_ratio",
        ],
    )


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        try:
            _ensure_not_locked("register", username)
            challenge = _verify_request_security("register")
            webcam_frames = _get_submission_inputs()
        except Exception as exc:
            engine.log_security_event("register", "blocked", username=username or None, ip_address=_client_address(), detail=str(exc))
            flash(str(exc), "error")
            return redirect(url_for("register"))
        if not username:
            flash("Username is required.", "error")
            return redirect(url_for("register"))
        if not webcam_frames:
            flash("Please use the webcam to capture a live registration session.", "error")
            return redirect(url_for("register"))
        try:
            result = engine.register_user(username, image_files=webcam_frames, challenge=challenge)
            engine.log_security_event("register", "success", username=username, ip_address=_client_address(), detail=str(result.get("blink")))
            _record_auth_outcome("register", True, username)
            flash(f"Registration completed for {username}.", "success")
            return render_template("result.html", result=result)
        except Exception as exc:
            engine.log_security_event("register", "failure", username=username, ip_address=_client_address(), detail=str(exc))
            _record_auth_outcome("register", False, username)
            app.logger.exception("Registration failed for '%s': %s", username, exc)
            flash(str(exc), "error")
            return redirect(url_for("register"))
    return _render_flow("register.html", "register")


@app.route("/authenticate", methods=["GET", "POST"])
def authenticate():
    if request.method == "POST":
        try:
            _ensure_not_locked("authenticate")
            challenge = _verify_request_security("authenticate")
            webcam_frames = _get_submission_inputs()
        except Exception as exc:
            engine.log_security_event("authenticate", "blocked", ip_address=_client_address(), detail=str(exc))
            flash(str(exc), "error")
            return redirect(url_for("authenticate"))
        if not webcam_frames:
            flash("Please use the webcam to capture a live authentication session.", "error")
            return redirect(url_for("authenticate"))
        try:
            result = engine.authenticate_user(image_files=webcam_frames, challenge=challenge)
            outcome = "success" if result.get("authenticated") else "failure"
            detail = f"challenge={challenge.get('action')}; comparison={result.get('comparison')}"
            engine.log_security_event("authenticate", outcome, username=result.get("username"), ip_address=_client_address(), detail=detail)
            _record_auth_outcome("authenticate", bool(result.get("authenticated")))
            return render_template("result.html", result=result)
        except Exception as exc:
            engine.log_security_event("authenticate", "failure", ip_address=_client_address(), detail=str(exc))
            _record_auth_outcome("authenticate", False)
            app.logger.exception("Authentication failed: %s", exc)
            flash(str(exc), "error")
            return redirect(url_for("authenticate"))
    return _render_flow("authenticate.html", "authenticate")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
