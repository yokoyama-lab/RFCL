from __future__ import annotations

from dataclasses import dataclass

from pyrev_fl.ast import ArrayRef, BinOp, Binary, Const, Expr, Place, UpdateOp, Var
from pyrev_fl.rl_ast import Assign, Block, Exit, FiFrom, FromEntry, FromLabel, Goto, IfGoto, Program, RFromLabel, RGoto, Swap


class ParseError(ValueError):
    pass


@dataclass(frozen=True)
class Token:
    text: str
    line: int
    column: int


def parse_program(source: str) -> Program:
    parser = _Parser(_tokenize(source))
    program = parser.parse_program()
    if parser.peek_text() is not None:
        parser.error(f"unexpected token: {parser.peek_text()}")
    return program


class _Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self.tokens = tokens
        self.pos = 0
        self.last_token: Token | None = None

    def parse_program(self) -> Program:
        return Program(
            inputs=self.parse_name_list(),
            outputs=self.parse_name_list(),
            temps=self.parse_name_list(),
            blocks=self.parse_blocks(),
        )

    def parse_name_list(self) -> list[str]:
        self.expect("(")
        names: list[str] = []
        while self.peek_text() != ")":
            names.append(self.parse_decl_name())
        self.expect(")")
        return names

    def parse_decl_name(self) -> str:
        name = self.next_required()
        if not _is_identifier(name):
            self.error(f"unexpected token: {name}")
        if self.peek_text() != "[":
            return name
        self.expect("[")
        raw_size = self.next_required()
        try:
            size = int(raw_size)
        except ValueError:
            self.error(f"unexpected token: {raw_size}")
        self.expect("]")
        return f"{name}[{size}]"

    def parse_blocks(self) -> list[Block]:
        blocks = []
        while self.peek_text() is not None:
            blocks.append(self.parse_block())
        return blocks

    def parse_block(self) -> Block:
        label = self.next_required()
        self.expect(":")
        from_ = self.parse_from()
        assigns = []
        while self.peek_text() not in {"goto", "rgoto", "if", "exit", None}:
            assigns.append(self.parse_assign())
        jump = self.parse_jump()
        return Block(label, from_, assigns, jump)

    def parse_from(self):
        token = self.peek_text()
        if token == "entry":
            self.expect("entry")
            self.expect(";")
            return FromEntry()
        if token == "from":
            self.expect("from")
            label = self.next_required()
            self.expect(";")
            return FromLabel(label)
        if token == "rfrom":
            self.expect("rfrom")
            label = self.next_required()
            self.expect(";")
            return RFromLabel(label)
        if token == "fi":
            self.expect("fi")
            expr = self.parse_expr()
            self.expect("from")
            true_label = self.next_required()
            self.expect("else")
            false_label = self.next_required()
            self.expect(";")
            return FiFrom(expr, true_label, false_label)
        self.error(f"unexpected token in from construct: {token}")

    def parse_jump(self):
        token = self.peek_text()
        if token == "goto":
            self.expect("goto")
            label = self.next_required()
            self.expect(";")
            return Goto(label)
        if token == "rgoto":
            self.expect("rgoto")
            label = self.next_required()
            self.expect(";")
            return RGoto(label)
        if token == "if":
            self.expect("if")
            expr = self.parse_expr()
            self.expect("goto")
            true_label = self.next_required()
            self.expect("else")
            false_label = self.next_required()
            self.expect(";")
            return IfGoto(expr, true_label, false_label)
        if token == "exit":
            self.expect("exit")
            self.expect(";")
            return Exit()
        self.error(f"unexpected token in jump construct: {token}")

    def parse_assign(self):
        left = self.parse_place()
        op = self.next_required()
        if op == "<=>":
            right = self.parse_place()
            self.expect(";")
            return Swap(left, right)

        updates = {
            "+=": UpdateOp.ADD,
            "-=": UpdateOp.SUB,
            "^=": UpdateOp.XOR,
        }
        try:
            update_op = updates[op]
        except KeyError as exc:
            self.error(f"unexpected token: {op}")

        expr = self.parse_expr()
        self.expect(";")
        return Assign(left, update_op, expr)

    def parse_place(self) -> str | Place:
        name = self.next_required()
        if not _is_identifier(name):
            self.error(f"unexpected token: {name}")
        return self._finish_ref(name)

    def parse_expr(self) -> Expr:
        token = self.peek_text()
        if token is None:
            self.error("unexpected end of input")
        if token == "(":
            self.expect("(")
            op = self.next_required()
            left = self.parse_expr()
            right = self.parse_expr()
            self.expect(")")
            return Binary(_parse_bin_op(op), left, right)
        raw = self.next_required()
        try:
            return Const(int(raw))
        except ValueError:
            if _is_identifier(raw):
                return self._finish_ref(raw)
            self.error(f"unexpected token: {raw}")

    def _finish_ref(self, name: str) -> Var | ArrayRef:
        if self.peek_text() != "[":
            return Var(name)
        self.expect("[")
        index = self.parse_expr()
        self.expect("]")
        return ArrayRef(name, index)

    def peek(self) -> Token | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def peek_text(self) -> str | None:
        token = self.peek()
        return None if token is None else token.text

    def expect(self, expected: str) -> None:
        found = self.next_required()
        if found != expected:
            self.error(f"expected {expected}, found {found}")

    def next_required(self) -> str:
        token = self.peek()
        if token is None:
            self.error("unexpected end of input")
        self.last_token = token
        self.pos += 1
        return token.text

    def error(self, message: str) -> None:
        token = self.last_token or self.peek()
        if token is None:
            raise ParseError(f"{message} at end of input")
        raise ParseError(f"{message} at line {token.line}, column {token.column}")


def _parse_bin_op(token: str) -> BinOp:
    for item in BinOp:
        if item.value == token:
            return item
    raise ParseError(f"unexpected token: {token}")


def _is_identifier(token: str) -> bool:
    if not token:
        return False
    if not (token[0].isalpha() or token[0] == "_"):
        return False
    return all(ch.isalnum() or ch in "_-" for ch in token[1:])


def _tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    current: list[str] = []
    i = 0
    line = 1
    column = 1
    start_column = 1
    multi = ["<=>", "+=", "-=", "^=", "!=", "<=", ">="]

    while i < len(source):
        ch = source[i]
        if ch.isspace():
            _flush(current, tokens, line, start_column)
            if ch == "\n":
                line += 1
                column = 1
            else:
                column += 1
            i += 1
            continue
        if ch == "#":
            _flush(current, tokens, line, start_column)
            while i < len(source) and source[i] != "\n":
                i += 1
                column += 1
            continue

        matched = next((op for op in multi if source.startswith(op, i)), None)
        if matched is not None:
            _flush(current, tokens, line, start_column)
            tokens.append(Token(matched, line, column))
            i += len(matched)
            column += len(matched)
            continue

        if ch in "();:[]":
            _flush(current, tokens, line, start_column)
            tokens.append(Token(ch, line, column))
            i += 1
            column += 1
            continue

        if not current:
            start_column = column
        current.append(ch)
        i += 1
        column += 1

    _flush(current, tokens, line, start_column)
    return tokens


def _flush(current: list[str], tokens: list[Token], line: int, column: int) -> None:
    if current:
        tokens.append(Token("".join(current), line, column))
        current.clear()
