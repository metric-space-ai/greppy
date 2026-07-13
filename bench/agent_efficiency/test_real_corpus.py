import re
import unittest

from bench.agent_efficiency import real_corpus


class RealCorpusContractTests(unittest.TestCase):
    def test_every_repository_uses_a_full_commit_sha(self) -> None:
        for name, spec in real_corpus.REPOS.items():
            with self.subTest(repository=name):
                self.assertRegex(spec["commit"], re.compile(r"^[0-9a-f]{40}$"))


if __name__ == "__main__":
    unittest.main()
