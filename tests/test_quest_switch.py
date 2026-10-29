"""QuestSwitch against a fake anvil-sw: what the glove's A button and a quiet glove do to the Quest switch.

The switch is what stands between a grip the controller holder keeps squeezed and the arm, so these pin the cases
that would leave an arm following: a press that does not land, a held button read as many presses, a glove gone
quiet that leaves its controller on, and a process that exits with its controllers on. Hardware- and network-free.
"""

import threading
import time

from prehensile.quest_switch import STALE_S, QuestSwitch


class FakeAnvil:
    """``GET``/``PUT /api/v1/quest/enabled`` as anvil-sw answers them, both controllers starting on."""

    def __init__(self):
        self.enabled = {"left": True, "right": True}
        self.changed = threading.Condition()

    def __call__(self, url, method, body=None):
        with self.changed:
            if method == "PUT":
                self.enabled = dict(body)
                self.changed.notify_all()
            return dict(self.enabled)

    def wait_for(self, expected):
        with self.changed:
            assert self.changed.wait_for(lambda: self.enabled == expected, timeout=2.0), self.enabled


def test_a_press_flips_its_side_once_however_long_it_is_held_and_exiting_switches_off():
    anvil = FakeAnvil()
    with QuestSwitch("http://anvil", ["left"], _http=anvil) as switch:
        assert anvil.enabled == {"left": False, "right": True}, "entering switches only its own side off"
        now = time.monotonic()
        for tick in range(5):  # one press held across five fresh frames
            switch.observe("left", fresh=True, a_button=True, now=now + tick * 0.01)
        anvil.wait_for({"left": True, "right": True})
        switch.observe("left", fresh=True, a_button=False, now=now + 0.1)
        switch.observe("left", fresh=True, a_button=True, now=now + 0.11)
        anvil.wait_for({"left": False, "right": True})
        switch.observe("left", fresh=True, a_button=False, now=now + 0.12)
        switch.observe("left", fresh=True, a_button=True, now=now + 0.13)
        anvil.wait_for({"left": True, "right": True})
    assert anvil.enabled == {"left": False, "right": True}, "leaving switches it off again"


def test_a_glove_gone_quiet_switches_its_controller_off():
    anvil = FakeAnvil()
    with QuestSwitch("http://anvil", ["left", "right"], _http=anvil) as switch:
        now = time.monotonic()
        switch.observe("right", fresh=True, a_button=True, now=now)
        anvil.wait_for({"left": False, "right": True})
        switch.observe("right", fresh=False, a_button=True, now=now + STALE_S / 2)
        assert anvil.enabled["right"], "a frame or two missed is not a quiet glove"
        switch.observe("right", fresh=False, a_button=True, now=now + STALE_S * 2)
        anvil.wait_for({"left": False, "right": False})
