import unittest
import pandas

#from dslminer import multiregression
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dslminer.multiregression import MultiRegression

class RegressionTest(unittest.TestCase):
    def setUp(self):
        self.indictor_id=61901
        self.orgunit_id=23519
        self.cadre_id = [33]
        self.multiregression=MultiRegression()
        self.multiregression.set_max_min_period(self.orgunit_id,self.indictor_id)

    # Test if indicator data frame created
    def test_indicator_data_frame(self):
        self.assertIsInstance(self.multiregression.get_indicator_data(self.orgunit_id,self.indictor_id),pandas.DataFrame)

    # Test if cadre data frame has values
    def test_cadre_data_frame(self):
        self.assertIsInstance(self.multiregression.get_cadres_by_year(self.orgunit_id, self.cadre_id),pandas.DataFrame)

    # Test if indicator data frame created
    def test_if_indicator_data_frame_emty(self):
        self.assertTrue(len(self.multiregression.get_indicator_data(self.orgunit_id, self.indictor_id))>3
                              )

    # Test if cadre data frame has values
    def test_if_cadre_data_frame_empy(self):
        self.assertTrue(len(self.multiregression.get_cadres_by_year(self.orgunit_id, self.cadre_id))>3
                              )



if __name__ == '__main__':
    unittest.main()