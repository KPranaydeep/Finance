import copy
import unittest
from public_release_checks import inspect_public_data,prepare_evidence_export


class EvidenceExportTests(unittest.TestCase):
    def fixture(self):
        row={"forecast_json":{"history_source":"DEVELOPMENT_BACKFILL"}}
        return {"nav":[{"is_backfill":True}],
                "forecasts":[copy.deepcopy(row),copy.deepcopy(row)],
                "active_forecasts":[copy.deepcopy(row),copy.deepcopy(row)]}

    def test_exact_provenance_allowed_and_source_unchanged(self):
        original=self.fixture()
        before=copy.deepcopy(original)
        export,findings=prepare_evidence_export(original)
        self.assertEqual(findings,[])
        self.assertEqual(export["evidence_metadata"]["classification"],"RESEARCH_SIMULATION")
        self.assertEqual(original,before)
        self.assertTrue(inspect_public_data(original,production=True))

    def test_no_backfill_means_no_exception(self):
        original=self.fixture()
        original["nav"]=[{"is_backfill":False}]
        _,findings=prepare_evidence_export(original)
        self.assertEqual(len(findings),4)
        original["nav"]=[{"is_backfill":"true"}]
        self.assertEqual(len(prepare_evidence_export(original)[1]),4)

    def test_secrets_and_private_paths_still_block(self):
        original=self.fixture()
        original["forecasts"][0]["forecast_json"].update(
            password="hidden",source="postgresql://private",
            location="C:\\Users\\Someone\\private.csv")
        _,findings=prepare_evidence_export(original)
        self.assertEqual(len(findings),3)
        self.assertTrue(all(item.startswith("Sensitive") for item in findings))

    def test_other_markers_and_paths_still_block(self):
        original=self.fixture()
        original["notes"]="DEVELOPMENT_BACKFILL"
        original["forecasts"][1]["forecast_json"]["history_source"]="DEVELOPMENT_OTHER"
        _,findings=prepare_evidence_export(original)
        self.assertEqual(findings,[
            "Non-production marker at $.forecasts[1].forecast_json.history_source",
            "Non-production marker at $.notes"])

    def test_live_evidence_stays_strict(self):
        export,findings=prepare_evidence_export({"nav":[{"is_backfill":False}],"forecasts":[]})
        self.assertEqual(findings,[])
        self.assertEqual(export["evidence_metadata"]["classification"],"POST_PUBLICATION_MODEL")


if __name__=="__main__": unittest.main()
