"""
resource_throttle.py — Best-effort CPU/RAM throttling for a training
subprocess.

There's no simple, safe, cross-platform way to hard-cap a Python
process's CPU/RAM from outside without risking crashing it mid-training
(especially RAM -- forcibly capping it can just turn into a hard crash
instead of a graceful slowdown). Instead this uses a duty-cycle approach:
watch the subprocess with psutil, and briefly suspend/resume it to bring
its average CPU usage back under the configured limit. RAM is monitored
and logged as a warning if it climbs past the configured limit, since
that's actionable information even without a hard cap.
"""
import threading, time

def start_throttle(pid, cpu_limit_pct=None, ram_limit_mb=None, log=print, poll_interval=1.0):
    """
    Starts a background daemon thread watching `pid`. Returns a
    threading.Event -- call .set() on it once the subprocess finishes to
    stop the throttle thread cleanly.
    """
    stop_event = threading.Event()
    if not cpu_limit_pct and not ram_limit_mb:
        return stop_event  # nothing configured to enforce

    try:
        import psutil
    except ImportError:
        log("Resource throttling unavailable (psutil not installed) -- CPU/RAM limits will not be enforced.")
        return stop_event

    def _loop():
        try:
            p = psutil.Process(pid)
        except Exception:
            return
        ncores = psutil.cpu_count() or 1
        p.cpu_percent(interval=None)  # prime psutil's internal counter
        warned_ram = False
        while not stop_event.is_set():
            time.sleep(poll_interval)
            try:
                if not p.is_running():
                    return
                if cpu_limit_pct:
                    # cpu_percent() can read up to 100*ncores; normalize
                    # to a single 0-100 scale so the slider means "percent
                    # of total machine capacity", matching the UI label.
                    cpu = p.cpu_percent(interval=None) / ncores
                    if cpu > cpu_limit_pct:
                        over = min(cpu / max(cpu_limit_pct, 1), 4.0)
                        p.suspend()
                        time.sleep(min(0.5 * over, 2.0))
                        if not stop_event.is_set():
                            p.resume()
                if ram_limit_mb:
                    mem_mb = p.memory_info().rss / (1024 * 1024)
                    if mem_mb > ram_limit_mb and not warned_ram:
                        log(f"WARNING: training is using {mem_mb:.0f}MB RAM, over your "
                            f"configured {ram_limit_mb}MB limit. Consider a smaller model, "
                            f"lower max length, or raising the RAM limit in Settings.")
                        warned_ram = True
            except Exception:
                # Process object can raise NoSuchProcess/AccessDenied if
                # the subprocess exits mid-check -- just stop quietly.
                return

    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return stop_event
