import unittest
from mysite.NaiveBayes import call_NB

class Testing(unittest.TestCase):
    def test_call_NB(self):
        data1 = 'jumanji 3'
        data2 = 'bore'
        result = call_NB(data1, data2)
        self.assertEqual(result[2],'1.2')

if __name__ == '__main__':
    unittest.main
