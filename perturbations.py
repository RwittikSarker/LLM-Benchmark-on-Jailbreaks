import random
import string


class Perturbation:
    def __init__(self, q):
        self.q = q
        self.alphabet = string.printable


class RandomSwapPerturbation(Perturbation):
    def __call__(self, s):
        list_s = list(s)
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))
        for i in sampled_indices:
            list_s[i] = random.choice(self.alphabet)
        return "".join(list_s)


class RandomPatchPerturbation(Perturbation):
    def __call__(self, s):
        list_s = list(s)
        substring_width = int(len(s) * self.q / 100)
        max_start = len(s) - substring_width
        start_index = random.randint(0, max_start)
        sampled_chars = "".join([random.choice(self.alphabet) for _ in range(substring_width)])
        list_s[start_index : start_index + substring_width] = sampled_chars
        return "".join(list_s)


class RandomInsertPerturbation(Perturbation):
    def __call__(self, s):
        list_s = list(s)
        sampled_indices = random.sample(range(len(s)), int(len(s) * self.q / 100))
        for i in sampled_indices:
            list_s.insert(i, random.choice(self.alphabet))
        return "".join(list_s)

