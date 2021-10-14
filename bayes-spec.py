import re
import bayes
import unittest

# n = bayes.BayesianNetwork(*edges, *vertices)


class TestParseLogic(unittest.TestCase):
    def test_shallow_joined(self):
        self.assertEqual(bayes.parse_query('P(A|B)='),
                         bayes.Query(bayes.Conditional(bayes.Node('A'), [bayes.Node('B')])))

    def test_shallow_negation(self):
        self.assertEqual(bayes.parse_query('P(~A)='), bayes.Query(bayes.Conditional(None, [
                         bayes.NegatedNode('A')])))

    def test_shallow_double_negation(self):
        self.assertEqual(bayes.parse_query('P(~~A)='),
                         bayes.Query(bayes.Conditional(None, [bayes.Node('A')])))

    def test_shallow_and(self):
        self.assertEqual(bayes.parse_query('P(A,B)='),
                         bayes.Query(bayes.Conditional(None, [bayes.Node('A'), bayes.Node('B')])))

    def test_deep_and(self):
        self.assertEqual(bayes.parse_query('P(A,B,C)='),
                         bayes.Query(bayes.Conditional(None, [bayes.Node('A'), bayes.Node('B'), bayes.Node('C')])))

    def test_shallow_and_and_joined(self):
        self.assertEqual(bayes.parse_query('P(A|B,C)='),
                         bayes.Query(bayes.Conditional(bayes.Node('A'), [bayes.Node('B'), bayes.Node('C')])))

    def test_deep_negation_and_and_joined(self):
        self.assertEqual(bayes.parse_query('P(~A|B,~C,D)='),
                         bayes.Query(bayes.Conditional(bayes.NegatedNode(('A')), [bayes.Node('B'),  bayes.NegatedNode('C'), bayes.Node('D')])))


def apply_probability_from_string(n, query):
    lexer = bayes.BayesLexer()
    parser = bayes.BayesParser()
    return bayes.apply_probability(n, parser.parse(lexer.tokenize(query)))


class TestParseProbability(unittest.TestCase):
    def setUp(self):
        self.n = bayes.BayesianNetwork(('R', 'S'), ('R', 'G'), ('S', 'G'))

    def testShallowParseProbability(self):
        apply_probability_from_string(self.n, 'P(S|R)=0.2')

        self.assertDictEqual(self.n.P['R'], {
            (True,): 0.2,
            (False,): 0.8
        })

    def testJointParseProbability(self):
        apply_probability_from_string(self.n, 'P(S|~R)=0.4')
        apply_probability_from_string(self.n, 'P(S|R)=0.01')

        self.assertDictEqual(self.n.P['S'], {
            (False, True): 0.4,
            (True, True): 0.01,
            (False, False): 0.6,
            (True, False): 0.99
        })

    def testJointParseProbabilityErrors(self):
        with self.assertRaises(ValueError):
            apply_probability_from_string(self.n, '(S|~R)=0.4')
        with self.assertRaises(ValueError):
            apply_probability_from_string(self.n, 'P(S)=0.01')
        with self.assertRaises(ValueError):
            apply_probability_from_string(self.n, 'P(S|R,S)=0.01')
        with self.assertRaises(ValueError):
            apply_probability_from_string(self.n, 'P(S|R,G)=0.01')

    def testJointParseProbability(self):
        apply_probability_from_string(self.n, 'P(G|~S,~R)=0.0')
        apply_probability_from_string(self.n, 'P(G|~S,R)=0.8')
        apply_probability_from_string(self.n, 'P(G|S,~R)=0.9')
        apply_probability_from_string(self.n, 'P(G|S,R)=0.99')

        self.assertAlmostEqual(self.n.P['G'][(True, False, False)], .0)
        self.assertAlmostEqual(self.n.P['G'][(True, True, False)], 0.8)
        self.assertAlmostEqual(self.n.P['G'][(True, False, True)], 0.9)
        self.assertAlmostEqual(self.n.P['G'][(True, True, True)], 0.99)
        self.assertAlmostEqual(self.n.P['G'][(False, False, False)], 1.0)
        self.assertAlmostEqual(self.n.P['G'][(False, True, False)], 0.2)
        self.assertAlmostEqual(self.n.P['G'][(False, False, True)], 0.1)
        self.assertAlmostEqual(self.n.P['G'][(False, True, True)], 0.01)
        self.assertAlmostEqual(self.n.P['G'][(True, False, False)], .0)
        # {
        #     (False, False, True): 0.0,
        #     (True, False, True): 0.8,
        #     (False, True, True): 0.9,
        #     (True, True, True): 0.99,
        #     (False, False, False): 1.0,
        #     (True, False, False): 0.2,
        #     (False, True, False): 0.1,
        #     (True, True, False): 0.01,
        # }


class TestBayesianNetwork(unittest.TestCase):
    def setUp(self):
        self.n = bayes.BayesianNetwork(('R', 'S'), ('R', 'G'), ('S', 'G'))

        apply_probability_from_string(self.n, 'P(R)=0.2')
        apply_probability_from_string(self.n, 'P(S|~R)=0.4')
        apply_probability_from_string(self.n, 'P(S|R)=0.01')
        apply_probability_from_string(self.n, 'P(G|~S,~R)=0.0')
        apply_probability_from_string(self.n, 'P(G|~S,R)=0.8')
        apply_probability_from_string(self.n, 'P(G|S,~R)=0.9')
        apply_probability_from_string(self.n, 'P(G|S,R)=0.99')

    def test_probability_no_event(self):
        self.assertAlmostEqual(self.n.probability(('R', True), {}), .2)
        self.assertAlmostEqual(self.n.probability(('R', False), {}), .8)

    def test_probability_event(self):
        self.assertAlmostEqual(self.n.probability(
            ('S', True), {"R": False}), .4)
        self.assertAlmostEqual(self.n.probability(
            ('S', False), {"R": False}), .6)
        self.assertAlmostEqual(self.n.probability(
            ('S', True), {"R": True}), .01)
        self.assertAlmostEqual(self.n.probability(
            ('S', False), {"R": True}), .99)

    def test_probability_deep_event(self):
        self.assertAlmostEqual(self.n.probability(
            ('G', True), {'S': False, "R": False}), 0)
        self.assertAlmostEqual(self.n.probability(
            ('G', True), {'S': False, "R": True}), .8)
        self.assertAlmostEqual(self.n.probability(
            ('G', True), {'S': True, "R": False}), .9)
        self.assertAlmostEqual(self.n.probability(
            ('G', True), {'S': True, "R": True}), .99)

    def test_predict(self):
        self.assertAlmostEqual(self.n.predict({'R': True}), 0.2)
        self.assertAlmostEqual(self.n.predict({'R': False}), 0.8)
        self.assertAlmostEqual(self.n.predict({'S': True}), 0.322)
        # self.assertAlmostEqual(self.n.predict({'G': True}), 0.404)
        self.assertAlmostEqual(self.n.predict(
            {'R': True, 'S': True, 'G': True}), 0.00198)

    def test_predict_inverse_robability_parts(self):
        self.assertAlmostEqual(self.n.predict(
            {'R': True, 'G': True}), 0.1584+0.00198)
        self.assertAlmostEqual(self.n.predict(
            {'G': True}), 0.00198+0.288+0.1584)

    def test_string_query(self):
        self.assertAlmostEqual(self.n.string_query('R'), 0.2)
        self.assertAlmostEqual(self.n.string_query('~R'), 0.8)
        self.assertAlmostEqual(self.n.string_query('S'), 0.322)
        # self.assertAlmostEqual(self.n.string_query({'G': True}), 0.404)
        self.assertAlmostEqual(self.n.string_query('R,S,G'), 0.00198)

    def test_string_query_inverse_robability_parts(self):
        self.assertAlmostEqual(self.n.string_query(
            'R,G'), 0.1584+0.00198)
        self.assertAlmostEqual(self.n.string_query(
            'G'), 0.00198+0.288+0.1584)


if __name__ == '__main__':
    unittest.main()
