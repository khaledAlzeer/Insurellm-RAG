from insurellm_rag.conversation import ConversationState
from insurellm_rag.schemas import ChatTurn


def state_after(question="Who is Avery Lancaster?", entities=("Avery Lancaster",)):
    s = ConversationState(max_turns=3)
    s.add_turn(ChatTurn(user=question, assistant="(unverified)", user_entities=list(entities),
                        sources=["Avery Lancaster"]))
    return s


def test_no_history_is_never_follow_up():
    assert not ConversationState().decide("What is her role?").is_follow_up


def test_pronoun_follow_up_resolves_to_previous_entity():
    s = state_after()
    d = s.decide("What is her role?")
    assert d.is_follow_up and d.focus[0] == "Avery Lancaster"
    assert s.resolve_deterministic("What is her role?", d) == "What is Avery Lancaster's role?"


def test_topic_shift_on_new_named_entity():
    s = state_after()
    d = s.decide("What does the Widget Pro product do?")
    assert not d.is_follow_up and d.reason == "new named entity"
    assert s.resolve_deterministic("What does the Widget Pro product do?", d) is None


def test_self_contained_question_is_not_follow_up():
    assert not state_after().decide("List the company's contract types").is_follow_up


def test_memory_is_bounded_and_resettable():
    s = state_after()
    for i in range(10):
        s.add_turn(ChatTurn(user=f"q{i}", assistant="a"))
    assert len(s.turns) == 3
    s.reset()
    assert not s.turns


def test_assistant_text_is_not_carried_into_focus():
    s = state_after()
    s.turns[-1].assistant = "She was born on Mars."  # unverified claim
    d = s.decide("What about her education?")
    assert all("Mars" not in f for f in d.focus)
