#!/usr/bin/python
# -*- coding: UTF-8 -*-
import rospy
import numpy as np
from pyuwds.tools.glove import GloveManager
from uwds_msgs.srv import SimpleQuery

np.random.seed(123)  # for reproducibility

class SparqlTranslater:
    def __init__(self):


if __name__ == '__main__':
    rospy.init_node("sparql_translater", anonymous=False)
    translater = SparqlTranslater()
    rospy.spin()
