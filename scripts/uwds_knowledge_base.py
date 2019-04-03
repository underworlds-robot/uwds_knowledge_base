#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from uwds_msgs.srv import QueryInContext
from pyuwds.uwds_client import UwdsClient
from pyuwds.uwds import UNDEFINED
from pyoro import Oro

class KnowledgeBase(UwdsClient):
    def __init__(self):
        UwdsClient.__init__("uwds_knowledge_base", UNDEFINED)
        hostname = rospy.get_param("~oro_host", "localhost")
        port = rospy.get_param("~oro_port", "6969")
        self.kb = Oro(hostname, port)
        self.query_service = rospy.Service("uwds/query_knowledge_base", QueryInContext, self.handleQuery)

    def addNode(self, world_name, node):
        namespace = world_name.split("/")
        agent = namespace[0]
        world = namespace[1]

        scene = ctx.worlds[world_name].scene

        types_str = scene.getNodeProperty(node.id, "class")
        if types_str != "" :
            types = types_str.split(",")
        else :
            if node.type == MESH : types.append("TangibleThing")
            if node.type == ENTITY : type.append("LocalizedThing")
            if node.type == CAMERA : type.append("ExistingThing")

        for type in types:
            if agent == "robot":
                self.kb.add(node.id+" rdf:Type "+property.data, world)
            else :
                self.kb.addForAgent(agent, node.id+" rdf:Type "+property.data, world)
        if agent == "robot":
            success = self.kb.checkConsistency()
        else :
            success = self.kb.checkForAgent(agent)
        return success

    def removeNode(self, world_name, node_id):
        pass

    def addSituation(self, world_name, situation):
        pass

    def removeSituation(self, world_name, situation):
        pass

    def queryKnowledgeBase(self, world_name, query):
        namespace = world_name.split("/")
        agent = namespace[0]
        world = namespace[1]
        if agent == "robot" :
            result = self.kb.find(world, query.split(","))
        else:
            result = self.kb.findForAgent(agent, world, query.split(","))
        pass

    def handleQuery(req):
        result = self.queryKnowledgeBase(req.ctxt.world, req.query)
        return result, True, ""

if __name__ == '__main__':
    kb = KnowledgeBase()
    rospy.spin()
