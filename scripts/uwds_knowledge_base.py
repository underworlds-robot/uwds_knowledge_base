#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from uwds_msgs.srv import QueryInContext
from pyuwds.uwds_client import UwdsClient
from pyuwds.uwds import READER
from pyuwds.types.nodes import CAMERA, MESH, ENTITY
from pyuwds.types.situations import FACT, ACTION, GENERIC, INTERNAL
from pyoro import Oro

class KnowledgeBase(UwdsClient):
    def __init__(self):
        UwdsClient.__init__(self, "uwds_knowledge_base", READER)
        hostname = rospy.get_param("~oro_host", "localhost")
        port = rospy.get_param("~oro_port", "6969")
        self.ontology_path = rospy.get_param("~ontology_path", "")
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

        self.__created_nodes = {}
        self.__created_situations = {}
        self.__support = {}
        self.__graspable = {}
        self.__container = {}

    def addNode(self, world_name, node):
        if world_name+node.id not in self.__created_nodes:
            namespace = world_name.split("/")
            agent = namespace[0]
            world = namespace[1]
            oro_agent = "myself" if agent == "robot" else agent
            types = []
            scene = self.ctx.worlds()[world_name].scene()

            types_str = scene.nodes().get_node_property(node.id, "class")
            if types_str != "":
                types = types_str.split(",")
            else:
                if node.type == MESH: types.append("TangibleThing")
                if node.type == ENTITY: types.append("LocalizedThing")
                if node.type == CAMERA: types.append("ExistingThing")

            seq = []
            for type in types:
                seq.append(node.id+" rdf:type "+type)
            seq.append(node.id+" rdfs:label "+node.name)

            self.kb.safeAddForAgent(oro_agent, seq)

            self.__created_nodes[world_name+node.id] = True
        return True

    def removeNode(self, world_name, node_id):
        """
        """
        pass

    def save(self):
        self.kb.save(self.ontology_path)

    def updateSituation(self, world_name, situation):
        """
        """
        success = False
        namespace = world_name.split("/")
        agent = namespace[0]
        world = namespace[1]
        oro_agent = "myself" if agent == "robot" else agent

        timeline = self.ctx.worlds()[world_name].timeline()

        subject = timeline.situations().get_situation_property(situation.id, "subject")
        object = timeline.situations().get_situation_property(situation.id, "object")


        if situation.type == ACTION:
            predicate = timeline.situations().get_situation_property(situation.id, "action")
        else:
            predicate = timeline.situations().get_situation_property(situation.id, "predicate")

        if object == "":
            if predicate == "Pick" or predicate == "Place":
                if subject not in self.__graspable:
                    self.kb.safeAddForAgent(oro_agent, [subject+" rdf:type GraspableObject"])
                    self.__graspable[subject] = True
            return True

        if predicate == "isOn":
            if object not in self.__support:
                self.kb.safeAddForAgent(oro_agent, [object+" rdf:type PhysicalSupport"])
                self.__support[object] = True

        if predicate == "isIn":
            if object not in self.__support:
                self.kb.safeAddForAgent(oro_agent, [object+" rdf:type Container"])
                self.__container[object] = True

        situation_str = ""
        if subject != "":
            if predicate != "":
                if object != "":
                    situation_str = subject+" "+predicate+" "+object

        if situation_str=="":
            return
        if situation.end.data == rospy.Time(0):
            if situation_str not in self.__created_situations:
                if situation != "":
                    if agent == "robot":
                        success = self.kb.safeAddForAgent("myself", [situation_str])
                    else:
                        success = self.kb.safeAddForAgent(agent, [situation_str])
                self.__created_situations[situation_str] = True
            else:
                return True
        else:
            if situation_str in self.__created_situations:
                if situation_str != "":
                    self.kb.removeForAgent(oro_agent, [situation_str])
                    success = True
                    del self.__created_situations[situation_str]
            else:
                return True
        return success

    def onChanges(self, world_name, header, invalidations):
        scene = self.ctx.worlds()[world_name].scene()
        timeline = self.ctx.worlds()[world_name].timeline()

        for node_id in invalidations.node_ids_updated:
            self.addNode(world_name, scene.nodes()[node_id])
        for node_id in invalidations.node_ids_deleted:
            self.removeNode(world_name, scene.nodes()[node_id])
        for situation_id in invalidations.situation_ids_updated:
            self.updateSituation(world_name, timeline.situations()[situation_id])

    def queryKnowledgeBase(self, world_name, query):
        """
        """
        if not self.ctx.worlds().has(world_name):
            scene = self.ctx.worlds()[world_name].scene()
            timeline = self.ctx.worlds()[world_name].timeline()
            self.ctx.worlds()[world_name].connect(self.onChanges)
            rospy.loginfo("nb nodes : "+str(len(scene.nodes()))+" (root included)")
            for node in scene.nodes():
                if node.name != "root":
                    self.addNode(world_name, node)
            rospy.loginfo("nb situations : "+str(len(timeline.situations())))
            for situation in timeline.situations():
                self.updateSituation(world_name, situation)
        namespace = world_name.split("/")
        agent = namespace[0]
        world = namespace[1]
        result = []
        if(self.verbose):
            rospy.loginfo("Query the <"+world_name+"> world : "+query)

        result_final = []
        oro_agent = "myself" if agent == "robot" else agent
        query_seq = query.split(",")
        if len(query_seq) > 1:
            results = self.kb.findForAgent(oro_agent, query_seq[0].split(" ")[0], query_seq[1:])
        else:
            results = self.kb.findForAgent(oro_agent, query_seq[0].split(" ")[0], query_seq)
        for result in results:
            if self.ctx.worlds()[world_name].scene().nodes().has(result):
                result_final.append(result)
            elif self.ctx.worlds()[world_name].timeline().situations().has(result):
                result_final.append(result)
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
