from __future__ import annotations
from typing import TypeAlias, Iterable
from pprint import pformat
from functools import reduce, cached_property
from operator import mul
from graphviz import Digraph

Symbol: TypeAlias = str|None

class NFAl:
  def __init__(self,
               delta: tuple[dict[Symbol, set[int]],...] = ({None: {1}}, {}),
               start: tuple[int,...] = (0,),
               final: tuple[int,...] = (1,)):
    nstate = len(delta)
    assert all(0 <= s < nstate for s in start), "incorrect start state"
    assert all(0 <= s < nstate for s in final), "incorrect final state"
    self.delta, self.start, self.final = delta, start, final

  @cached_property
  def syms(self) -> tuple[str, ...]: return tuple(sorted({s for tr in self.delta for s in tr if s is not None}))

  @staticmethod
  def symbols(s: Symbol|Iterable[Symbol]) -> NFAl:
    syms = (None,) if s is None else tuple(s)
    delta = (*({syms[i]: {i+1}} for i in range(len(syms))), {})
    return NFAl(delta, (0,), (len(delta)-1,))

  def concat(self, s: NFAl|Symbol|Iterable[Symbol]) -> NFAl:
    o = s if isinstance(s, NFAl) else NFAl.symbols(s)
    off = len(self.delta) # offset <-> second start

    # offset second NFAl states
    odelta = tuple({k: {x+off for x in v} for k, v in s.items()} for s in o.delta)
    delta: tuple[dict[Symbol, set[int]],...] = (*self.delta, *odelta)
    final = tuple(x+off for x in o.final)

    # bridge first with second NFAl
    for f in self.final: delta[f].setdefault(None, set()).add(off)

    return NFAl(delta, self.start, final)

  def pow(self, p: int) -> NFAl: return reduce(mul, [self] * max(0, p), NFAl())

  def union(self, s: NFAl|Iterable[Symbol]|Symbol) -> NFAl:
    o = s if isinstance(s, NFAl) else NFAl.symbols(s)
    off1, off2 = 1, len(self.delta) + 1
    delta1 = tuple({k: {x+off1 for x in v} for k, v in s.items()} for s in self.delta)
    delta2 = tuple({k: {x+off2 for x in v} for k, v in s.items()} for s in o.delta)
    final = len(delta1) + len(delta2) + 1 # 2 -> 1 start + 1 final; and minus 1 because of 0-based indexing
    for f in self.final: delta1[f].setdefault(None, set()).add(final)
    for f in o.final: delta2[f].setdefault(None, set()).add(final)
    delta: tuple[dict[Symbol, set[int]], ...] = ({None: {off1, off2}}, *delta1, *delta2, {})
    return NFAl(delta, (0,), (final,))

  def star(self) -> NFAl:
    off = 1
    idelta = tuple({k: {x+off for x in v} for k, v in s.items()} for s in self.delta)
    delta: tuple[dict[Symbol, set[int]], ...] = ({None: {off}}, *idelta, {None: {0}})
    final = len(delta)-1
    for f in self.final: delta[f+off].setdefault(None, set()).add(final)
    return NFAl(delta, (0,), (final,))

  def plus(self) -> NFAl: return self * self.star()

  def __mul__(self, other: NFAl | Symbol) -> NFAl: return self.concat(other)
  def __or__(self, other: NFAl | Symbol) -> NFAl: return self.union(other)
  def __pow__(self, p: int) -> NFAl: return self.pow(p)

  def __repr__(self): return str(self)
  def __str__(self):
    return "\n".join((
      f"delta: {pformat(self.delta, indent=2, width=100)}",
      f"start: {pformat(self.start, indent=2, width=100)}",
      f"final: {pformat(self.final, indent=2, width=100)}"))

  def graphviz(self):
    g = Digraph(graph_attr={"rankdir":"LR"})
    g.node("start", shape="point", width="0.1", height="0.1", label="")
    for i in range(len(self.delta)):
      attrs = {"label": str(i), "shape": "circle"}
      if i in self.final: attrs["shape"] = "doublecircle"
      g.node(str(i), **attrs)
      if i in self.start: g.edge("start", str(i))
    for i, state in enumerate(self.delta):
      for sym, transitions in state.items():
        for state_next in transitions:
          lbl = str(sym if sym is not None else "λ")
          g.edge(str(i), str(state_next), label=lbl)
    g.render(format="png", view=True)



ab = NFAl.symbols("ab")
cd = NFAl.symbols("cd")

# abcd = ab * cd
# print(abcd)
# abcd.graphviz()

# ab_cd = ab | cd
# print(ab_cd)
# ab_cd.graphviz()

# a3 = NFAl.symbols("a").pow(3)
# print(a3)
# a3.graphviz()

# astar = NFAl.symbols("a").star()
# astar.graphviz()

aplus = NFAl.symbols("a").plus()
# aplus.graphviz()
