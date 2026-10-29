"""QuestSwitch: the glove's A button as the on/off switch of an anvil-sw Quest controller.

With the Quest controller strapped to the glove, its holder keeps the side grip
squeezed, so the grip can no longer be the arm's clutch. anvil-sw lets each
controller be switched off through ``PUT /api/v1/quest/enabled``: off, a
controller drives nothing however far its grip is squeezed; switched on with
the grip held, it takes its arm from where the arm stands. This drives that
switch from the glove:

  * a press of a side's A button flips that side's controller (left glove ->
    left controller), as anvil-sw has it at that moment, so a switch flipped on
    the anvil-sw page is never undone behind the operator's back;
  * a glove that goes quiet -- no fresh frame for that side for ``STALE_S`` --
    switches its controller off: the glove's buttons are only ever read off a
    fresh frame, so a dead glove would otherwise leave its arm following;
  * starting switches both of this process's controllers off, so a grip the
    holder keeps squeezed cannot take an arm before the operator presses A, and
    stopping switches them off again.

The HTTP calls run on a thread of their own, so a slow or absent backend never
stalls the hand loop; a call that fails is retried every ``RETRY_S`` and said
each time.
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.request

ROUTE = "/api/v1/quest/enabled"
# How long a side's glove may go without a fresh frame before its controller is
# switched off: a few frames at HandDriver's 120 Hz, well under a human reaction.
STALE_S = 0.25
RETRY_S = 0.5
TIMEOUT_S = 1.0


class QuestSwitchError(RuntimeError):
    """anvil-sw's Quest switch could not be read or written."""


def _http(url: str, method: str, body: dict | None = None) -> dict:
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        return json.loads(response.read())


class QuestSwitch:
    """Drives anvil-sw's Quest controller switch for ``sides`` from the gloves.

    Use as a context manager around the hand loop and call ``observe`` once per
    side per tick. Entering switches every side off and raises
    ``QuestSwitchError`` if anvil-sw cannot be reached, before anything moves;
    leaving switches them off again. ``_http`` is a test seam.
    """

    def __init__(self, base_url: str, sides, _http=_http) -> None:
        self._url = base_url.rstrip("/") + ROUTE
        self._sides = tuple(sides)
        self._call = _http
        self._lock = threading.Lock()
        # What each side still owes anvil-sw: None, "flip" or "off".
        self._pending: dict[str, str | None] = dict.fromkeys(self._sides)
        self._fresh_at: dict[str, float | None] = dict.fromkeys(self._sides)
        # Quiet until its first frame: starting has already switched it off.
        self._quiet = dict.fromkeys(self._sides, True)
        self._previous_a = dict.fromkeys(self._sides, False)
        self._wake = threading.Event()
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "QuestSwitch":
        self._write(dict.fromkeys(self._sides, "off"))
        print(f"quest switch: {', '.join(self._sides)} controller(s) OFF; press the glove's A "
              f"button to switch one on ({self._url})", flush=True)
        self._thread = threading.Thread(target=self._run, name="quest-switch", daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc) -> bool:
        self._stopped.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join()
        try:
            self._write(dict.fromkeys(self._sides, "off"))
            print(f"\nquest switch: {', '.join(self._sides)} controller(s) OFF", flush=True)
        except QuestSwitchError as error:
            print(f"\nquest switch: could not switch the controllers off on the way out ({error}); "
                  "switch them off on the anvil-sw page", file=sys.stderr, flush=True)
        return False

    def observe(self, side: str, fresh: bool, a_button: bool, now: float) -> None:
        """One tick of ``side``'s glove: ``fresh`` says a new frame arrived, carrying ``a_button``."""
        with self._lock:
            if fresh:
                self._fresh_at[side] = now
                self._quiet[side] = False
                if a_button and not self._previous_a[side]:
                    # Two presses before the first reached anvil-sw cancel out.
                    self._pending[side] = None if self._pending[side] == "flip" else "flip"
                    self._wake.set()
                self._previous_a[side] = a_button
                return
            fresh_at = self._fresh_at[side]
            if not self._quiet[side] and fresh_at is not None and now - fresh_at > STALE_S:
                self._quiet[side] = True
                self._pending[side] = "off"
                print(f"\nquest switch: the {side} glove went quiet; switching its controller off",
                      flush=True)
                self._wake.set()

    def _run(self) -> None:
        while not self._stopped.is_set():
            self._wake.wait(timeout=RETRY_S)
            self._wake.clear()
            if self._stopped.is_set():
                return
            with self._lock:
                owed = {side: op for side, op in self._pending.items() if op is not None}
            if not owed:
                continue
            try:
                written = self._write(owed)
            except QuestSwitchError as error:
                print(f"\nquest switch: could not reach anvil-sw ({error}); retrying every "
                      f"{RETRY_S:g} s", file=sys.stderr, flush=True)
                continue
            with self._lock:
                for side, op in owed.items():
                    # A press that landed while this call was out is still owed.
                    if self._pending[side] == op:
                        self._pending[side] = None
            for side in owed:
                print(f"\nquest switch: {side} controller {'ON' if written[side] else 'OFF'}",
                      flush=True)

    def _write(self, ops: dict[str, str]) -> dict:
        """Applies ``ops`` to the switch as anvil-sw has it now, leaving other sides alone."""
        try:
            switch = self._call(self._url, "GET")
            body = {"left": bool(switch["left"]), "right": bool(switch["right"])}
            for side, op in ops.items():
                body[side] = op == "flip" and not body[side]
            return self._call(self._url, "PUT", body)
        except (OSError, ValueError, KeyError) as error:  # URLError and timeouts are OSErrors
            raise QuestSwitchError(f"{self._url}: {error}") from error
