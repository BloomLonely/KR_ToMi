import numpy as np
from .kr_oracle import Oracle
from typing import List, Tuple
from pyjosa.josa import Josa

class Action(object):
    def __init__(self, templates):
        self.templates = templates

    def render(self):
        raise NotImplementedError

class DeclarativeAction(Action):
    def render(self):
        if hasattr(self, "fixed"):
            return self.templates[self.fixed]
        return np.random.choice(self.templates)

class InterrogativeAction(Action):
    def render(self):
        if hasattr(self, "fixed"):
            return self.templates[self.fixed]
        return np.random.choice(self.templates)

class SearchedAction(InterrogativeAction):
    def __init__(self, oracle: Oracle, agent: str, obj: str):
        ans = oracle.get_direct_belief(agent, obj)
        # Label whether or not this question requires theory of mind
        self.tom = ans != oracle.get_object_container(obj)

        agent_with_josa = Josa.get_full_string(agent, "은")
        obj_with_josa = Josa.get_full_string(obj, "을")

        question = f"{agent_with_josa} {obj_with_josa} 어디에서 찾을까?\t{ans}\t1"
        super().__init__([question])


class BeliefSearchAction(InterrogativeAction):
    def __init__(self, oracle: Oracle, a1: str, a2: str, obj: str):
        ans = oracle.get_indirect_belief(a1, a2, obj)
        # Does this question require theory of mind?
        self.tom = ans != oracle.get_object_container(obj)

        a1_with_josa = Josa.get_full_string(a1, "은")
        a2_with_josa = Josa.get_full_string(a2, "이가")
        obj_with_josa = Josa.get_full_string(obj, "을")

        question = f"{a1_with_josa} {a2_with_josa} {obj_with_josa} 어디에서 찾을 거라고 생각할까?\t{ans}\t1"
        super().__init__([question])


class RealityAction(InterrogativeAction):
    def __init__(self, oracle: Oracle, obj: str):
        ans = oracle.get_object_container(obj)

        obj_with_josa = Josa.get_full_string(obj, "은")

        question = f"{obj_with_josa} 실제로 어디에 있을까?\t{ans}\t1"
        super().__init__([question])


class MemoryAction(InterrogativeAction):
    def __init__(self, oracle_start_state: Oracle, obj: str):
        ans = oracle_start_state.locations.obj_containers[obj]

        obj_with_josa = Josa.get_full_string(obj, "은")

        question = f"{obj_with_josa} 처음에 어디에 있었을까?\t{ans}\t1"
        super().__init__([question])


class LocationAction(DeclarativeAction):
    def __init__(self, oracle: Oracle, args: str):
        if len(args) == 2:
            a1, loc = args

            a1_with_josa = Josa.get_full_string(a1, "은")
            statement = f"{a1_with_josa} {loc}에 있다."
            oracle.set_location(a1, loc)
        else:  # 2 people
            a1, a2, loc = args

            a1_with_josa = Josa.get_full_string(a1, "와")
            a2_with_josa = Josa.get_full_string(a2, "은")
            statement = f"{a1_with_josa} {a2_with_josa} {loc}에 있다."
            oracle.set_location(a1, loc)
            oracle.set_location(a2, loc)
        super().__init__([statement])


class ObjectLocAction(DeclarativeAction):
    def __init__(self, oracle: Oracle, obj: str, observers: List[str]):
        container = oracle.get_object_container(obj)

        obj_with_josa = Josa.get_full_string(obj, "은")

        statement = f"{obj_with_josa} {container}에 있다."
        super().__init__([statement])

        # set direct beliefs
        for observer in observers:
            oracle.set_direct_belief(observer, obj, container)

        # set indirect beliefs
        for observer1 in observers:
            for observer2 in observers:
                if observer1 != observer2:
                    oracle.set_indirect_belief(observer1, observer2, obj, container)


class ExitedAction(DeclarativeAction):
    def __init__(self, oracle: Oracle, agent: str):
        loc = oracle.get_location(agent)

        agent_with_josa = Josa.get_full_string(agent, "이가")

        statement = f"{agent_with_josa} {loc}에서 나갔다."
        super().__init__([statement])
        oracle.set_location(agent, None)


class MoveAction(DeclarativeAction):
    def __init__(
        self, oracle: Oracle, args: Tuple[str, str, str], observers: List[str] = None
    ):
        agent, obj, container = args

        agent_with_josa = Josa.get_full_string(agent, "이가")
        obj_with_josa = Josa.get_full_string(obj, "을")
        container_with_josa = Josa.get_full_string(container, "으로")

        statement = f"{agent_with_josa} {obj_with_josa} {container_with_josa} 옮겼다."
        super().__init__([statement])

        oracle.set_object_container(obj, container)

        if not observers:
            observers = []
        observers.append(agent)
        # set direct beliefs
        for observer in observers:
            oracle.set_direct_belief(observer, obj, container)

        # set indirect beliefs
        for observer1 in observers:
            for observer2 in observers:
                if observer1 != observer2:
                    oracle.set_indirect_belief(observer1, observer2, obj, container)


class PeekAction(DeclarativeAction):
    def __init__(self, oracle, args: Tuple[str, str], observers: List[str] = None):
        agent, container = args

        agent_with_josa = Josa.get_full_string(agent, "이가")

        statement = f"{agent_with_josa} {container}를 들여다보았다."
        super().__init__([statement])

        contents = oracle.get_container_obj(container)

        if not observers:
            observers = []

        observers.append(agent)
        # set direct beliefs
        for observer in observers:
            for obj in contents:
                oracle.set_direct_belief(observer, obj, container)

        # set indirect beliefs
        for observer1 in observers:
            for observer2 in observers:
                if observer1 != observer2:
                    for obj in contents:
                        oracle.set_indirect_belief(observer1, observer2, obj, container)


class TellAction(DeclarativeAction):
    def __init__(self, oracle: Oracle, a1: str, a2: str, obj: str):

        a1_with_josa = Josa.get_full_string(a1, "이가")
        a2_with_josa = Josa.get_full_string(a2, "에게")
        obj_with_josa = Josa.get_full_string(obj, "이가")

        statement = f"{a1_with_josa} {a2_with_josa} {obj_with_josa} 어디에 있는지 말했다."
        super().__init__([statement])

        container = oracle.get_object_container(obj)
        oracle.set_direct_belief(a2, obj, container)
        oracle.set_indirect_belief(a2, a1, obj, container)


class EnterAction(DeclarativeAction):
    def __init__(
        self,
        oracle: Oracle,
        args: Tuple[str, str],
        observers: List[str] = None,
        no_world_adjust: bool = False,
    ):
        agent, location = args

        agent_with_josa = Josa.get_full_string(agent, "이가")

        statement = f"{agent_with_josa} {location}에 들어갔다."
        super().__init__([statement])

        oracle.set_location(agent, location)
        # assume all containers are not enclosed
        # agent knows location of everything
        objs = oracle.get_objects_at_location(location)
        if not observers:
            observers = []
        observers.append(agent)

        if not no_world_adjust:
            for obj in objs:
                container = oracle.get_object_container(obj)
                oracle.set_direct_belief(agent, obj, container)
                for observer1 in observers:
                    for observer2 in observers:
                        if observer1 != observer2:
                            oracle.set_indirect_belief(
                                observer1, observer2, obj, container
                            )


class NoiseAction(DeclarativeAction):
    def __init__(self, oracle: Oracle, person: str, thing: str):

        person_with_josa = Josa.get_full_string(person, "은")
        thing_with_josa = Josa.get_full_string(thing, "을")

        templates = [
            f"{person_with_josa} {thing_with_josa} 좋아한다.",
            f"{person_with_josa} {thing_with_josa} 싫어한다.",
            f"{person_with_josa} {thing_with_josa} 매우 좋아한다.",
            f"{person_with_josa} {thing_with_josa} 정말 싫어한다.",
        ]
        super().__init__(templates)
        self.fixed = np.random.randint(0, len(self.templates))
