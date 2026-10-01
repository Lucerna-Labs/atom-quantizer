import unittest
from experiments.six_bit_frontier import final_gate,select


class QualityFrontierTests(unittest.TestCase):
    def fixture(self):
        names=("source-f32","historical-budget120","six-scalar","six-diagonal","six-activation")
        models={n:{"archive_bytes":1000 if n=="historical-budget120" else 800} for n in names}
        scores={n:{"status":"MEASURED","perplexity":30.0 if n=="historical-budget120" else 28.0} for n in names}
        admitted={n:True for n in names if n.startswith("six-")}
        return models,scores,admitted

    def test_global_choice_uses_eligibility_then_declared_ties(self):
        models,scores,admitted=self.fixture()
        self.assertEqual(select(models,scores,admitted)["selected"],"six-scalar")
        models["six-diagonal"]["archive_bytes"]=790
        self.assertEqual(select(models,scores,admitted)["selected"],"six-diagonal")
        scores["six-activation"]["perplexity"]=27.5
        self.assertEqual(select(models,scores,admitted)["selected"],"six-activation")
        admitted["six-activation"]=False
        self.assertEqual(select(models,scores,admitted)["selected"],"six-diagonal")
        models["six-diagonal"]["archive_bytes"]=901
        self.assertEqual(select(models,scores,admitted)["selected"],"six-scalar")
        scores["six-scalar"]["perplexity"]=29.2
        self.assertIsNone(select(models,scores,admitted)["selected"])

    def test_final_never_reselects_from_a_better_unchosen_result(self):
        models,scores,_=self.fixture()
        scores["six-activation"]["perplexity"]=29.2
        self.assertEqual(final_gate("six-activation",models,scores)["status"],"FAILED")
        self.assertEqual(final_gate("six-activation",models,scores)["selected"],"six-activation")
        scores["six-activation"]["perplexity"]=28
        self.assertEqual(final_gate("six-activation",models,scores)["status"],"QUALIFIED")
        scores["six-scalar"]["status"]="FAILED"
        self.assertEqual(final_gate("six-activation",models,scores)["status"],"INCONCLUSIVE")


if __name__=="__main__": unittest.main()
