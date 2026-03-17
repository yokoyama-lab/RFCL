"""Janus → SRL parser.

Parses a simplified Janus subset that corresponds to SRL:

    procedure main()
        int var1
        int var2
        ...
        <statements>

Supported statements (no semicolons):
    x += expr
    x -= expr
    x ^= expr
    x <=> y
    if expr then ... else ... fi expr
    from expr do ... loop ... until expr
    skip

Expressions: infix with standard operator precedence.
Operators: =  !=  <  <=  >  >=  +  -  *  /

The SRL interface (inputs / outputs / temps) is recovered from the
``// SRL inputs=(...) outputs=(...) temps=(...)`` comment written by
``render_janus``.  If the comment is absent, all declared variables are
placed in both inputs and outputs (identity interface).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from pyrev_fl.ast import Assign, BinOp, Binary, Block, Const, Expr, If, Loop, Program, Swap, UpdateOp, Var


class JanusParseError(ValueError):
    pass


@dataclass(frozen=True)
class _Token:
    text: str
    line: int
    col: int


def parse_janus(source: str) -> Program:
    """Parse a Janus ``procedure main()`` and return the equivalent SRL Program."""
    interface = _extract_srl_interface(source)
    tokens = _tokenize(source)
    parser = _JanusParser(tokens)
    declared_vars, body = parser.parse_main()

    if interface is not None:
        inputs, outputs, temps = interface
    else:
        # All declared variables are both inputs and outputs
        inputs = declared_vars
        outputs = declared_vars
        temps = []

    return Program(inputs=inputs, outputs=outputs, temps=temps, body=body)


# ---------------------------------------------------------------------------
# Interface comment extraction
# ---------------------------------------------------------------------------

_INTERFACE_RE = re.compile(
    r"//\s*SRL\s+inputs=\(([^)]*)\)\s+outputs=\(([^)]*)\)\s+temps=\(([^)]*)\)"
)


def _extract_srl_interface(source: str) -> tuple[list[str], list[str], list[str]] | None:
    m = _INTERFACE_RE.search(source)
    if m is None:
        return None
    def _names(s: str) -> list[str]:
        return [t for t in s.split() if t]
    return _names(m.group(1)), _names(m.group(2)), _names(m.group(3))


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

class _JanusParser:
    def __init__(self, tokens: list[_Token]) -> None:
        self._tokens = tokens
        self._pos = 0
        self._last: _Token | None = None

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def parse_main(self) -> tuple[list[str], Block]:
        """Parse ``procedure main()`` and return (declared_vars, body)."""
        # Skip to "procedure main()" - skip comments and other procedures
        self._skip_to_main()
        self._expect("procedure")
        self._expect("main")
        self._expect("(")
        self._expect(")")

        declared: list[str] = []
        # Parse variable declarations
        while self._peek() == "int":
            self._advance()  # consume "int"
            name = self._next_ident()
            declared.append(name)
            # Optional initializer: = value (skip it — SRL starts vars at 0)
            if self._peek() == "=":
                self._advance()
                self._parse_expr()  # consume and discard

        body = self._parse_block(stop={"procedure", None})
        return declared, body

    def _skip_to_main(self) -> None:
        """Advance past any non-main procedures."""
        while True:
            tok = self._peek()
            if tok is None:
                break
            if tok == "procedure":
                saved = self._pos
                self._advance()
                if self._peek() == "main":
                    self._pos = saved  # rewind; let parse_main consume it
                    return
                # Skip this procedure's body: advance until next "procedure" or EOF
                depth = 0
                while self._peek() is not None:
                    t = self._advance()
                    if t == "procedure":
                        self._pos -= 1
                        break
            else:
                self._advance()

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def _parse_block(self, stop: set[str | None]) -> Block:
        stmts = []
        while self._peek() not in stop:
            tok = self._peek()
            if tok is None:
                break
            stmts.append(self._parse_stmt())
        return Block(stmts)

    def _parse_stmt(self):
        tok = self._peek()
        if tok == "skip":
            self._advance()
            return None  # skip → nothing; filtered below
        if tok == "if":
            return self._parse_if()
        if tok == "from":
            return self._parse_loop()
        if tok is None:
            self._error("unexpected end of input")
        return self._parse_assign_or_swap()

    def _parse_block_filtered(self, stop: set[str | None]) -> Block:
        """Like _parse_block but filters out None (skip) results."""
        stmts = []
        while self._peek() not in stop:
            tok = self._peek()
            if tok is None:
                break
            s = self._parse_stmt()
            if s is not None:
                stmts.append(s)
        return Block(stmts)

    def _parse_if(self) -> If:
        self._expect("if")
        test = self._parse_expr()
        self._expect("then")
        then_block = self._parse_block_filtered({"else"})
        self._expect("else")
        else_block = self._parse_block_filtered({"fi"})
        self._expect("fi")
        assertion = self._parse_expr()
        return If(test, then_block, else_block, assertion)

    def _parse_loop(self) -> Loop:
        self._expect("from")
        entry_guard = self._parse_expr()
        self._expect("do")
        do_block = self._parse_block_filtered({"loop"})
        self._expect("loop")
        loop_block = self._parse_block_filtered({"until"})
        self._expect("until")
        exit_guard = self._parse_expr()
        return Loop(entry_guard, do_block, loop_block, exit_guard)

    def _parse_assign_or_swap(self):
        name = self._next_ident()
        op = self._advance()
        if op == "<=>":
            right = self._next_ident()
            return Swap(name, right)
        updates = {"+=": UpdateOp.ADD, "-=": UpdateOp.SUB, "^=": UpdateOp.XOR}
        if op not in updates:
            self._error(f"expected assignment operator or <=>, got {op!r}")
        expr = self._parse_expr()
        return Assign(name, updates[op], expr)

    # ------------------------------------------------------------------
    # Expressions (infix Pratt-style, 3 precedence levels)
    # ------------------------------------------------------------------

    def _parse_expr(self) -> Expr:
        return self._parse_comparison()

    def _parse_comparison(self) -> Expr:
        left = self._parse_additive()
        op = self._peek()
        if op in {"=", "!=", "<", "<=", ">", ">="}:
            self._advance()
            right = self._parse_additive()
            return Binary(_janus_op(op), left, right)
        return left

    def _parse_additive(self) -> Expr:
        left = self._parse_multiplicative()
        while self._peek() in {"+", "-"}:
            op = self._advance()
            right = self._parse_multiplicative()
            left = Binary(_janus_op(op), left, right)
        return left

    def _parse_multiplicative(self) -> Expr:
        left = self._parse_atom()
        while self._peek() in {"*", "/"}:
            op = self._advance()
            right = self._parse_atom()
            left = Binary(_janus_op(op), left, right)
        return left

    def _parse_atom(self) -> Expr:
        tok = self._advance()
        if tok is None:
            self._error("unexpected end of input in expression")
        if tok == "(":
            expr = self._parse_expr()
            self._expect(")")
            return expr
        try:
            return Const(int(tok))
        except ValueError:
            pass
        if _is_ident(tok):
            return Var(tok)
        self._error(f"unexpected token in expression: {tok!r}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _peek(self) -> str | None:
        if self._pos >= len(self._tokens):
            return None
        return self._tokens[self._pos].text

    def _advance(self) -> str:
        tok = self._tokens[self._pos]
        self._last = tok
        self._pos += 1
        return tok.text

    def _next_ident(self) -> str:
        tok = self._advance()
        if not _is_ident(tok):
            self._error(f"expected identifier, got {tok!r}")
        return tok

    def _expect(self, expected: str) -> None:
        got = self._advance()
        if got != expected:
            self._error(f"expected {expected!r}, got {got!r}")

    def _error(self, msg: str) -> None:
        tok = self._last or (self._tokens[self._pos] if self._pos < len(self._tokens) else None)
        if tok:
            raise JanusParseError(f"{msg} at line {tok.line}, col {tok.col}")
        raise JanusParseError(msg)


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

_MULTI = ["<=>", "+=", "-=", "^=", "!=", "<=", ">="]
_SINGLES = set("()[]{}=+-*/")


def _tokenize(source: str) -> list[_Token]:
    tokens: list[_Token] = []
    buf: list[str] = []
    i = 0
    line = 1
    col = 1
    buf_col = 1

    def flush():
        if buf:
            tokens.append(_Token("".join(buf), line, buf_col))
            buf.clear()

    while i < len(source):
        ch = source[i]

        # Line comments
        if ch == "/" and source[i:i+2] == "//":
            flush()
            while i < len(source) and source[i] != "\n":
                i += 1
                col += 1
            continue

        if ch.isspace():
            flush()
            if ch == "\n":
                line += 1
                col = 1
            else:
                col += 1
            i += 1
            continue

        # Multi-char operators
        matched = next((op for op in _MULTI if source.startswith(op, i)), None)
        if matched:
            flush()
            tokens.append(_Token(matched, line, col))
            col += len(matched)
            i += len(matched)
            continue

        # Single-char delimiters that always become their own token
        if ch in _SINGLES:
            flush()
            tokens.append(_Token(ch, line, col))
            col += 1
            i += 1
            continue

        # Accumulate into buffer
        if not buf:
            buf_col = col
        buf.append(ch)
        col += 1
        i += 1

    flush()
    return tokens


def _is_ident(token: str) -> bool:
    if not token:
        return False
    if not (token[0].isalpha() or token[0] == "_"):
        return False
    return all(ch.isalnum() or ch in "_-" for ch in token[1:])


_JANUS_OPS: dict[str, BinOp] = {
    "+": BinOp.ADD, "-": BinOp.SUB, "*": BinOp.MUL, "/": BinOp.DIV,
    "=": BinOp.EQ, "!=": BinOp.NE,
    "<": BinOp.LT, "<=": BinOp.LE, ">": BinOp.GT, ">=": BinOp.GE,
}


def _janus_op(token: str) -> BinOp:
    try:
        return _JANUS_OPS[token]
    except KeyError as exc:
        raise JanusParseError(f"unknown operator: {token!r}") from exc
