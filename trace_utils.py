from datetime import datetime, timezone
import os
import threading


def trace_event(event, **fields):
    """Emit one structured control-flow line for Render logs."""
    timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
    details = " ".join(
        f"{key}={_compact(value)}" for key, value in fields.items()
    )
    print(
        f"[TRACE] ts={timestamp} pid={os.getpid()} "
        f"thread={threading.current_thread().name} event={event} {details}",
        flush=True,
    )


def _compact(value):
    text = repr(value)
    if len(text) > 300:
        return text[:297] + "..."
    return text
