"""Undo/Redo（コマンドパターン）。Qt非依存でテスト可能."""
from __future__ import annotations

from typing import Callable, List, Optional


class Command:
    label = "操作"

    def do(self):
        raise NotImplementedError

    def undo(self):
        raise NotImplementedError


class FuncCommand(Command):
    def __init__(self, label: str, do: Callable, undo: Callable):
        self.label = label
        self._do = do
        self._undo = undo

    def do(self):
        self._do()

    def undo(self):
        self._undo()


class CompositeCommand(Command):
    def __init__(self, label: str, commands: List[Command]):
        self.label = label
        self.commands = commands

    def do(self):
        for c in self.commands:
            c.do()

    def undo(self):
        for c in reversed(self.commands):
            c.undo()


class UndoStack:
    def __init__(self, max_depth: int = 50):
        self._undo: List[Command] = []
        self._redo: List[Command] = []
        self.max_depth = max_depth
        self.listeners: List[Callable[[], None]] = []
        self.clean_index = 0

    def push(self, cmd: Command, execute: bool = True):
        if execute:
            cmd.do()
        self._undo.append(cmd)
        self._redo.clear()
        if len(self._undo) > self.max_depth:
            self._undo.pop(0)
            self.clean_index -= 1
        self._notify()

    def undo(self) -> Optional[str]:
        if not self._undo:
            return None
        cmd = self._undo.pop()
        cmd.undo()
        self._redo.append(cmd)
        self._notify()
        return cmd.label

    def redo(self) -> Optional[str]:
        if not self._redo:
            return None
        cmd = self._redo.pop()
        cmd.do()
        self._undo.append(cmd)
        self._notify()
        return cmd.label

    @property
    def can_undo(self):
        return bool(self._undo)

    @property
    def can_redo(self):
        return bool(self._redo)

    @property
    def undo_label(self):
        return self._undo[-1].label if self._undo else ""

    @property
    def redo_label(self):
        return self._redo[-1].label if self._redo else ""

    def clear(self):
        self._undo.clear()
        self._redo.clear()
        self.clean_index = 0
        self._notify()

    def set_clean(self):
        self.clean_index = len(self._undo)

    @property
    def is_clean(self):
        return self.clean_index == len(self._undo)

    def _notify(self):
        for f in list(self.listeners):
            f()
