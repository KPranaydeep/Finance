import copy
import unittest
from decimal import Decimal
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

    def review_fees(self, amount=20.):
        return {'nav': [{'is_backfill': False}], 'review_events': [
            {'kind': 'BASELINE', 'payload': {'lots': [
                {'entry_charges': {'brokerage': amount}}]}},
            {'kind': 'ASSESSMENT', 'payload': {'metrics': {'rows': [
                {'fees': {'brokerage': amount}}]}}},
        ]}

    def test_model_fee_amounts_are_public_and_source_is_preserved(self):
        for amount in (0, 20, 3.25, Decimal('9.99')):
            with self.subTest(amount=amount):
                original = self.review_fees(amount)
                before = copy.deepcopy(original)
                evidence, findings = prepare_evidence_export(original)
                self.assertEqual(findings, [])
                self.assertEqual(original, before)
                self.assertEqual(evidence['review_events'], original['review_events'])
                self.assertEqual(inspect_public_data(original), [])

    def test_brokerage_exception_does_not_allow_non_amount_values(self):
        for value in ('20.00', 'postgresql://private', '/home/private/file',
                      {'password': 'private'}, True, -1, float('nan'),
                      float('inf'), Decimal('NaN'), 10 ** 1000):
            with self.subTest(value_type=type(value).__name__):
                _, findings = prepare_evidence_export(self.review_fees(value))
                self.assertTrue(any('Sensitive field name' in f for f in findings))

    def test_unknown_brokerage_locations_still_block(self):
        for value in ({'brokerage': 20.}, {'fees': {'brokerage': 20.}},
                      {'review_events': [{'payload': {'brokerage': 20.}}]}):
            self.assertTrue(inspect_public_data(value))

    def test_private_fields_inside_valid_fee_containers_still_block(self):
        value = self.review_fees()
        fees = value['review_events'][1]['payload']['metrics']['rows'][0]['fees']
        fees.update(brokerage_password='private', broker_account_number='private',
                    api_key='private', database_url='postgresql://private')
        _, findings = prepare_evidence_export(value)
        self.assertTrue(any('.brokerage_password' in f for f in findings))
        self.assertTrue(any('.broker_account_number' in f for f in findings))
        self.assertTrue(any('.api_key' in f for f in findings))
        self.assertTrue(any('.database_url' in f for f in findings))
        self.assertFalse(any(f.endswith('.fees.brokerage') for f in findings))


if __name__=="__main__": unittest.main()
