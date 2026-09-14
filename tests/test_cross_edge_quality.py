import unittest

from hela_mem.analyze_cross_edge_quality import build_records, extract_cross_edges, parse_response, summarize


class CrossEdgeQualityTest(unittest.TestCase):
    def test_extracts_only_cross_edges(self):
        graph = {"edges": {"0": {"1": 0.5, "2": 0.2}, "1": {"0": 0.5}, "2": {"0": 0.2, "3": 0.6}, "3": {"2": 0.6}}}
        self.assertEqual(extract_cross_edges(graph, ["0", "1"]), [{"base_id": "0", "nonbase_id": "2", "edge_weight": 0.2}])

    def test_builds_target_once_for_multiple_edges(self):
        item = {"question_id": "q", "question": "Who?", "answer": "Ada", "question_type": "single"}
        prediction = {"retrieved_episodic": [{"source": "base", "node_id": "0"}, {"source": "base", "node_id": "1"}]}
        graph = {"nodes": {"0": {"content": "base a"}, "1": {"content": "base b"}, "2": {"content": "Ada did it"}}, "edges": {"0": {"2": 0.2}, "1": {"2": 0.3}}}
        record = build_records(item, prediction, graph, 15)
        self.assertEqual(len(record["cross_edges"]), 2)
        self.assertEqual(len(record["candidate_targets"]), 1)
        self.assertEqual(record["candidate_targets"][0]["connected_base_ids"], ["0", "1"])

    def test_parses_only_valid_labels(self):
        result = parse_response('{"labels":[{"candidate_id":"2","label":"supporting","confidence":1.2,"rationale":"fact"},{"candidate_id":"x","label":"irrelevant"}]}', {"2"})
        self.assertEqual(result["2"]["label"], "supporting")
        self.assertEqual(result["2"]["confidence"], 1.0)

    def test_summarizes_question_and_edge_rates(self):
        records = [{"question_id": "q", "cross_edges": [{"edge_weight": 0.2}], "annotated_cross_edges": [{"label": "supporting", "edge_weight": 0.2}], "candidate_annotations": [{"label": "supporting"}]}]
        report = summarize(records, dataset_question_count=5)
        self.assertEqual(report["questions_with_supporting_target"], 1)
        self.assertEqual(report["edge_label_rates"]["supporting"], 1.0)
        self.assertEqual(report["questions_with_cross_edges_rate"], 0.2)


if __name__ == "__main__":
    unittest.main()
