import json
import unittest
from public_nav_snapshots import encode_snapshot,load_nav_snapshot,save_nav_snapshot


class Result:
    def __init__(self,row): self.row=row
    def fetchone(self): return self.row


class Store:
    def __init__(self):
        self.exists=False
        self.snapshots=[]
    def execute(self,query,params=()):
        if "to_regclass" in query:
            return Result({"relation":"public_model_nav_snapshots" if self.exists else None})
        if "CREATE TABLE" in query: self.exists=True
        if "INSERT INTO public_model_nav_snapshots" in query:
            basket,version,encoded,digest=params
            self.snapshots.append(dict(basket=basket,nav_rows=json.loads(encoded),payload_sha256=digest))
        if "SELECT nav_rows" in query:
            matching=[r for r in self.snapshots if r["basket"]==params[0]]
            return Result(matching[-1] if matching else None)
        return Result(None)


class SnapshotTests(unittest.TestCase):
    def test_shorter_rerun_does_not_retain_old_dates_or_levels(self):
        conn=Store()
        save_nav_snapshot(conn,"A",[dict(nav_date="2024-01-01",nav=100),
                                   dict(nav_date="2024-01-02",nav=500)],6)
        save_nav_snapshot(conn,"A",[dict(nav_date="2024-01-02",nav=100)],6)
        rows=load_nav_snapshot(conn,"A")
        self.assertEqual(len(rows),1)
        self.assertEqual(rows[0]["nav"],100)
        self.assertEqual(str(rows[0]["nav_date"]),"2024-01-02")
        self.assertEqual(len(conn.snapshots),2)
        self.assertEqual(conn.snapshots[0]["nav_rows"][1]["nav"],500)

    def test_basket_isolation_and_missing_history(self):
        conn=Store()
        self.assertEqual(load_nav_snapshot(conn,"A"),[])
        save_nav_snapshot(conn,"B",[dict(nav_date="2024-01-01",nav=100)],6)
        self.assertEqual(load_nav_snapshot(conn,"A"),[])

    def test_tampering_is_rejected(self):
        conn=Store()
        save_nav_snapshot(conn,"A",[dict(nav_date="2024-01-01",nav=100)],6)
        conn.snapshots[0]["nav_rows"][0]["nav"]=999
        with self.assertRaises(ValueError):
            load_nav_snapshot(conn,"A")

    def test_invalid_histories_cannot_be_published(self):
        for rows in ([],[dict(nav_date="2024-01-01",nav=0)],
                     [dict(nav_date="2024-01-01",nav=float("nan"))],
                     [dict(nav_date="2024-01-02",nav=100),dict(nav_date="2024-01-01",nav=100)]):
            with self.assertRaises(ValueError):
                encode_snapshot(rows)


if __name__=="__main__": unittest.main()
