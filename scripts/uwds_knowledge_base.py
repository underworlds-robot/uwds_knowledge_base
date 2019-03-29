#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from pyuwds.uwds import UwdsClient, UNDEFINED
from pyoro import Oro

class KnowledgeBase(UwdsClient):
    def __init__(self, hostname, port):
        UwdsClient.__init__("uwds_knowledge_base", UNDEFINED)
        self.kb = Oro(hostname, port)

    def addIndividual(self, world_name, node):
        for property in node.properties:
            if property.name == "class":
                self.kb += [node.id+" rdf:Type "+property.data]
                return True
        return False

    def removeIndividual(self, world_name, node_id):
        for property in ctx.worlds[node_id].properties:
            if property.name == "class":
                self.kb -= [node.id+" rdf:Type "+property.data]
                return True
        return False

    def addProperty(self, world_name, situation):
        pass

    def removeProperty(self, world_name, situation):
        pass

    def queryKnowledgeBase(self, world_name, string):
        pass

if __name__ == '__main__':
    kb = KnowledgeBase("localhost", 6969)
    rospy.spin()
