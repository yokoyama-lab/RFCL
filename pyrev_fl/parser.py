from __future__ import annotations

from dataclasses import dataclass

from pyrev_fl.ast import ArrayRef, Assign, BinOp, Binary, Block, Const, Expr, If, Loop, Place, Pop, Program, Push, Rif, Swap, UpdateOp, Var


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
    if parser.peek() is not None:
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
            body=self.parse_block_until(set()),
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
            # Check for stack declaration: "name:stack"
            if ":stack" in name:
                base = name.replace(":stack", "")
                if _is_identifier(base):
                    return f"{base}:stack"
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

    def parse_block_until(self, stop: set[str]) -> Block:
        stmts = []
        while self.peek_text() is not None and self.peek_text() not in stop:
            stmts.append(self.parse_stmt())
        return Block(stmts)

    def parse_stmt(self):
        token = self.peek_text()
        if token == "if":
            return self.parse_if()
        if token == "from":
            return self.parse_loop()
        if token == "rif":
            return self.parse_rif()
        if token == "push":
            return self.parse_push()
        if token == "pop":
            return self.parse_pop()
        if token is None:
            self.error("unexpected end of input")
        return self.parse_simple_stmt()

    def parse_push(self) -> Push:
        self.expect("push")
        var = self.next_required()
        stack = self.next_required()
        self.expect(";")
        return Push(var, stack)

    def parse_pop(self) -> Pop:
        self.expect("pop")
        var = self.next_required()
        stack = self.next_required()
        self.expect(";")
        return Pop(var, stack)

    def parse_if(self) -> If:
        self.expect("if")
        test = self.parse_expr()
        self.expect("then")
        then_block = self.parse_block_until({"else"})
        self.expect("else")
        else_block = self.parse_block_until({"fi"})
        self.expect("fi")
        assertion = self.parse_expr()
        return If(test, then_block, else_block, assertion)

    def parse_loop(self) -> Loop:
        self.expect("from")
        entry_guard = self.parse_expr()
        self.expect("do")
        do_block = self.parse_block_until({"loop"})
        self.expect("loop")
        loop_block = self.parse_block_until({"until"})
        self.expect("until")
        exit_guard = self.parse_expr()
        return Loop(entry_guard, do_block, loop_block, exit_guard)

    def parse_rif(self) -> Rif:
        self.expect("rif")
        test = self.parse_expr()
        body = self.parse_block_until({"rfi"})
        self.expect("rfi")
        assertion = self.parse_expr()
        return Rif(test, body, assertion)

    def parse_simple_stmt(self):
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
            raise ParseError(f"unexpected token: {op}") from exc

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
    mapping = {item.value: item for item in BinOp}
    try:
        return mapping[token]
    except KeyError as exc:
        raise ParseError(f"unexpected token: {token}") from exc


def _is_identifier(token: str) -> bool:
    if not token:
        return False
    if not (token[0].isalpha() or token[0] == "_"):
        return False
    return all(ch.isalnum() or ch in "_-" for ch in token[1:])


def _tokenize(source: str) -> list[Token]:
    tokens: list[Token] = []
    current = []
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

        if ch in "();[]":
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
