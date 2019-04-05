#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from uwds_msgs.srv import QueryInContext
from pyuwds.uwds_client import UwdsClient
from pyuwds.uwds import READER
from pyoro import Oro

class KnowledgeBase(UwdsClient):
    def __init__(self):
        UwdsClient.__init__(self, "uwds_knowledge_base", READER)
        hostname = rospy.get_param("~oro_host", "localhost")
        port = rospy.get_param("~oro_port", "6969")
        success = False
        while not success and not rospy.is_shutdown():
            try:
                self.kb = Oro(hostname, int(port))
                success = True
            except Exception as e:
                pass
        rospy.loginfo("Connected to the Oro knowledge base")
        self.query_service = rospy.Service("uwds/query_knowledge_base", QueryInContext, self.handleQuery)
        rospy.loginfo("Underworlds KB ready !")

    def addNode(self, world_name, node):
        namespace = world_name.split("/")
        agent = namespace[0]
        world = namespace[1]

        scene = self.ctx.worlds[world_name].scene

        types_str = scene.getNodeProperty(node.id, "class")
        if types_str != "":
            types = types_str.split(",")
        else:
            if node.type == MESH: types.append("TangibleThing")
            if node.type == ENTITY: type.append("LocalizedThing")
            if node.type == CAMERA: type.append("ExistingThing")

        for type in types:
            if agent == "robot":
                self.kb.add(node.id+" rdf:Type "+property.data, world)
            else:
                self.kb.addForAgent(agent, node.id+" rdf:Type "+property.data, world)
        if agent == "robot":
            success = self.kb.checkConsistency()
        else:
            success = self.kb.checkForAgent(agent)
        return success

    def removeNode(self, world_name, node_id):
        """
        """
        pass

    def addSituation(self, world_name, situation):
        """
        """
        namespace = world_name.split("/")
        agent = namespace[0]
        world = namespace[1]
        pass

    def removeSituation(self, world_name, situation):
        """
        """
        pass

    def queryKnowledgeBase(self, world_name, query):
        """
        """
        namespace = world_name.split("/")
        agent = namespace[0]
        world = namespace[1]
        result = []
        if(self.verbose):
            rospy.loginfo("Query the <"+world_name+"> world : "+query)

            if agent == "robot":
                result = self.kb.find(world, query.split(","))
            else:
                result = self.kb.findForAgent(agent, world, query.split(","))

        result_final = []
        for element in result:
            if self.ctx.worlds[world_name].scene.nodes.has(element):
                result_final.append(element)
            elif self.ctx.worlds[world_name].timeline.situations.has(element):
                result_final.append(element)
        return result_final

    def handleQuery(self, req):
        """
        """
        try:
            result = self.queryKnowledgeBase(req.ctxt.world, req.query)
            return result, True, ""
        except Exception as e:
            rospy.logwarn("[uwds::queryKnowledgeBase] Exception occurred : "+str(e))
            return [], False, str(e)


if __name__ == '__main__':
    rospy.init_node("uwds_knowledge_base", anonymous=False)
    kb = KnowledgeBase()
    rospy.spin()
