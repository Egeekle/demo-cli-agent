from src.agent import appointment_agent


def test_agent_registers_all_tools():
    names = set(appointment_agent._function_toolset.tools.keys())
    expected = {
        "list_appointments", "search_appointments", "get_appointment",
        "check_availability", "create_appointment", "update_appointment",
        "delete_appointment", "resolve_scheduling_conflicts",
    }
    assert expected <= names


def test_destructive_tools_require_approval():
    tools = appointment_agent._function_toolset.tools
    for name in ["create_appointment", "update_appointment", "delete_appointment"]:
        assert tools[name].requires_approval is True
