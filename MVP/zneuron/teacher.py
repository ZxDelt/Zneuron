"""A teacher that says words and rewards the right move, so the brain can learn simple word -> action links.

Lesson: hear "left" -> turn the head left; hear "right" -> turn the head right.
The reward comes the moment the head has clearly turned, because learning only stays eligible for ~1 s (#30).
"""

import random

LESSONS = {"left": ("head turn", +1), "right": ("head turn", -1)}


class Teacher:
    def __init__(self, agent, seed=0, words=("left", "right"), timeout_ms=1500, gap_ms=600, turn=0.12,
                 reward=1.0, punish=-0.6):
        self.agent, self.words = agent, list(words)
        self.rng = random.Random(seed)
        self.timeout_ms, self.gap_ms, self.turn = timeout_ms, gap_ms, turn
        self.reward, self.punish = reward, punish
        self.results = []                  # (t, word, outcome) with outcome "right" / "wrong" / "none"

    def trial(self):
        """One lesson: say a word, watch for a clear head turn, reward or punish it at once."""
        a = self.agent
        w, net = a.world, a.net
        word = self.rng.choice(self.words)
        joint, want = LESSONS[word]
        w.hear(word, net.t)
        start = w.joints[joint]
        outcome = "none"
        waited = 0
        while waited < self.timeout_ms:
            a.run(20)
            waited += 20
            moved = (w.joints[joint] - start) * want
            if moved > self.turn:
                outcome = "right"
                net.reward(self.reward)
                break
            if moved < -self.turn:
                outcome = "wrong"
                net.reward(self.punish)
                break
        self.results.append((net.t, word, outcome))
        a.run(self.gap_ms)
        return outcome

    def score(self, last=50):
        """Share of decided trials (head clearly turned) that went the right way, over the last `last` trials."""
        recent = [o for _, _, o in self.results[-last:] if o != "none"]
        return (sum(o == "right" for o in recent) / len(recent)) if recent else None
