"""InstructionManager and builtin instruction unit tests — 16 tests."""
from __future__ import annotations

from neuralmem.instructions.builtins import (
    BuiltinInstruction,
    apply_builtin,
    get_builtin,
    list_builtins,
)
from neuralmem.instructions.manager import (
    Instruction,
    InstructionManager,
    InstructionScope,
)

# ==================== InstructionScope ====================


class TestInstructionScope:

    def test_instruction_scope_enum(self):
        assert InstructionScope.GLOBAL.value == "global"
        assert InstructionScope.USER.value == "user"
        assert InstructionScope.AGENT.value == "agent"
        assert InstructionScope.SESSION.value == "session"
        assert len(InstructionScope) == 4


# ==================== Instruction dataclass ====================


class TestInstructionDataclass:

    def test_instruction_dataclass(self):
        inst = Instruction(
            id="test-id",
            name="test",
            content="do something",
            scope=InstructionScope.GLOBAL,
        )
        assert inst.id == "test-id"
        assert inst.name == "test"
        assert inst.content == "do something"
        assert inst.scope is InstructionScope.GLOBAL
        assert inst.scope_id is None
        assert inst.priority == 0
        assert inst.enabled is True
        assert inst.created_at is not None


# ==================== InstructionManager ====================


class TestInstructionManager:

    def test_instruction_manager_add(self):
        mgr = InstructionManager()
        inst = mgr.add(
            name="test_rule",
            content="Always extract in English",
            scope=InstructionScope.GLOBAL,
        )
        assert isinstance(inst, Instruction)
        assert inst.name == "test_rule"
        assert inst.content == "Always extract in English"
        assert len(inst.id) > 0

    def test_instruction_manager_remove(self):
        mgr = InstructionManager()
        inst = mgr.add(name="x", content="y")
        assert mgr.remove(inst.id) is True
        assert mgr.get(inst.id) is None
        # Removing again returns False
        assert mgr.remove(inst.id) is False

    def test_instruction_manager_get(self):
        mgr = InstructionManager()
        inst = mgr.add(name="x", content="y")
        assert mgr.get(inst.id) is inst
        assert mgr.get("nonexistent") is None

    def test_instruction_manager_list_all(self):
        mgr = InstructionManager()
        mgr.add(name="a", content="alpha")
        mgr.add(name="b", content="beta")
        all_items = mgr.list()
        assert len(all_items) == 2

    def test_instruction_manager_list_by_scope(self):
        mgr = InstructionManager()
        mgr.add(name="g", content="global", scope=InstructionScope.GLOBAL)
        mgr.add(name="u", content="user", scope=InstructionScope.USER)
        mgr.add(
            name="u2",
            content="user2",
            scope=InstructionScope.USER,
            scope_id="user-1",
        )
        globals_ = mgr.list(scope=InstructionScope.GLOBAL)
        assert len(globals_) == 1
        users = mgr.list(scope=InstructionScope.USER)
        assert len(users) == 2
        user_1 = mgr.list(
            scope=InstructionScope.USER, scope_id="user-1"
        )
        assert len(user_1) == 1

    def test_instruction_manager_get_prompt_injection(self):
        mgr = InstructionManager()
        mgr.add(name="a", content="Rule A")
        mgr.add(name="b", content="Rule B")
        prompt = mgr.get_prompt_injection()
        assert "Instructions:" in prompt
        assert "Rule A" in prompt
        assert "Rule B" in prompt

    def test_instruction_manager_get_prompt_injection_priority_order(self):
        mgr = InstructionManager()
        mgr.add(name="low", content="Low priority", priority=1)
        mgr.add(name="high", content="High priority", priority=10)
        prompt = mgr.get_prompt_injection()
        # High priority comes first
        high_pos = prompt.index("High priority")
        low_pos = prompt.index("Low priority")
        assert high_pos < low_pos

    def test_instruction_manager_clear(self):
        mgr = InstructionManager()
        mgr.add(name="a", content="x", scope=InstructionScope.GLOBAL)
        mgr.add(name="b", content="y", scope=InstructionScope.USER)
        count = mgr.clear(scope=InstructionScope.GLOBAL)
        assert count == 1
        assert len(mgr.list()) == 1
        count = mgr.clear()
        assert count == 1
        assert len(mgr.list()) == 0

    def test_instruction_manager_to_dict_from_dict_roundtrip(self):
        mgr = InstructionManager()
        mgr.add(
            name="rule1",
            content="Extract in English",
            scope=InstructionScope.GLOBAL,
            priority=5,
        )
        mgr.add(
            name="rule2",
            content="Be concise",
            scope=InstructionScope.USER,
            scope_id="u1",
        )
        data = mgr.to_dict()
        assert "instructions" in data
        assert len(data["instructions"]) == 2

        restored = InstructionManager.from_dict(data)
        items = restored.list()
        assert len(items) == 2
        names = {i.name for i in items}
        assert names == {"rule1", "rule2"}

    def test_instruction_prompt_injection_empty(self):
        mgr = InstructionManager()
        assert mgr.get_prompt_injection() == ""

    def test_instruction_disabled_excluded_from_prompt(self):
        mgr = InstructionManager()
        inst = mgr.add(name="x", content="Disabled rule")
        inst.enabled = False
        # Need to manually disable since add returns a new one
        mgr._instructions[inst.id].enabled = False
        prompt = mgr.get_prompt_injection()
        assert "Disabled rule" not in prompt


# ==================== Built-in instructions ====================


class TestBuiltinInstructions:

    def test_builtin_instruction_list(self):
        builtins = list_builtins()
        assert len(builtins) == 8
        assert all(isinstance(b, BuiltinInstruction) for b in builtins)

    def test_builtin_get_by_name(self):
        bi = get_builtin("language_en")
        assert bi is not None
        assert bi.name == "language_en"
        assert "English" in bi.content

        assert get_builtin("nonexistent") is None

    def test_builtin_apply_to_manager(self):
        mgr = InstructionManager()
        inst = apply_builtin(mgr, "privacy_mode")
        assert inst is not None
        assert inst.name == "privacy_mode"
        assert len(mgr.list()) == 1

    def test_builtin_apply_nonexistent(self):
        mgr = InstructionManager()
        result = apply_builtin(mgr, "no_such_builtin")
        assert result is None
        assert len(mgr.list()) == 0
