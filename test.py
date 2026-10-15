from __future__ import annotations
from functools import reduce, cached_property
from collections.abc import Iterable, Callable
from graphviz import Digraph
from pprint import pformat
import operator as op
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

type Symbol = str | None

def _mask(*v: int) -> int: return reduce(lambda a, b: a | (1 << b), v, 0)
def _unmask(b: int) -> set[int]: return {i for i in range(b.bit_length()) if b & (1 << i)}
def _cast[T,C](v: T | Iterable[T], c: Callable[[Iterable[T]], C]) -> C: return c(v) if isinstance(v, Iterable) else c((v,))

class DFA:
  __slots__ = ("_frozen", "delta", "entry", "final", "syms")
  def __init__(self, delta: list[list[int]], entry: int, final: int | Iterable[int], syms: Iterable[Symbol]):
    object.__setattr__(self, "delta", delta)
    object.__setattr__(self, "entry", entry)
    object.__setattr__(self, "final", final)
    object.__setattr__(self, "syms", syms)
    object.__setattr__(self, "_frozen", True)

  def __setattr__(self, name, value):
    if getattr(self, "_frozen", False): raise AttributeError("DFA is immutable")
    object.__setattr__(self, name, value)

  def graphviz(self):
    g = Digraph(graph_attr={"rankdir":"LR"})
    g.node("entry", shape="point", width="0.1", height="0.1", label="")
    for i in range(len(self.delta)):
      attrs = {"label": str(i), "shape": "circle"}
      if i in self.final: attrs["shape"] = "doublecircle"
      g.node(str(i), **attrs)
      if i == self.entry: g.edge("entry", str(i))
    for i, trans in enumerate(self.delta):
      t = {}
      for sym, dst in zip(self.syms, trans, strict=True): t.setdefault(dst, []).append(sym)
      for dst, s in t.items(): g.edge(str(i), str(dst), label=",".join(s))
    path = g.render(filename="graph", format="png")
    plt.imshow(mpimg.imread(path))
    plt.axis("off")
    plt.show(block=True)
    plt.close()

# TODO: NFAl should be immutable and a singleton (how do we compare two NFAls?)
class NFA:
  def __init__(self, delta: Iterable[dict[Symbol, set[int]]] = ({None: {1}}, {}), entry: int | Iterable[int] = 0, final: int | Iterable[int] = 1):
    delta, entry, final = tuple(delta), _cast(entry, frozenset), _cast(final, frozenset)
    assert all(0 <= s < len(delta) for s in entry), f"incorrect entry states: 0 <= {entry} < {len(delta)}"
    assert all(0 <= s < len(delta) for s in final), f"incorrect final states: 0 <= {final} < {len(delta)}"
    self.delta, self.entry, self.final = delta, entry, final

  @cached_property
  def syms(self) -> tuple[str, ...]: return tuple(sorted({s for tr in self.delta for s in tr if s is not None}))

  @staticmethod
  def symbols(s: Symbol|Iterable[Symbol]) -> NFA:
    syms = tuple(s or [None])
    delta = (*({syms[i]: {i+1}} for i in range(len(syms))), {})
    return NFA(delta, 0, len(delta)-1)

  @staticmethod
  def _to_nfa(s: NFA|Symbol|Iterable[Symbol]) -> NFA: return s if isinstance(s, NFA) else NFA.symbols(s)

  def concat(self, s: NFA|Symbol|Iterable[Symbol]) -> NFA:
    o = NFA._to_nfa(s)
    off = len(self.delta) # offset <-> second entry
    # offset second NFA states
    odelta = tuple({k: {x+off for x in v} for k, v in s.items()} for s in o.delta)
    final = tuple(x+off for x in o.final)
    # bridge first with second NFA (replace bridged states: never mutate self)
    delta = [*self.delta, *odelta]
    for f in self.final: delta[f] = {**delta[f], None: delta[f].get(None, set()) | {off}}
    return NFA(delta, self.entry, final)

  def pow(self, p: int) -> NFA: return reduce(op.mul, [self] * max(0, p), NFA())

  def union(self, s: NFA|Iterable[Symbol]|Symbol) -> NFA:
    o = NFA._to_nfa(s)
    off1, off2 = 1, len(self.delta) + 1
    delta1 = tuple({k: {x+off1 for x in v} for k, v in s.items()} for s in self.delta)
    delta2 = tuple({k: {x+off2 for x in v} for k, v in s.items()} for s in o.delta)
    final = len(delta1) + len(delta2) + 1 # 2 -> 1 entry + 1 final; and minus 1 because of 0-based indexing
    for f in self.final: delta1[f].setdefault(None, set()).add(final)
    for f in o.final: delta2[f].setdefault(None, set()).add(final)
    delta: tuple[dict[Symbol, set[int]], ...] = ({None: {off1, off2}}, *delta1, *delta2, {})
    return NFA(delta, (0,), (final,))

  def star(self) -> NFA:
    off = 1
    idelta = tuple({k: {x+off for x in v} for k, v in s.items()} for s in self.delta)
    delta: tuple[dict[Symbol, set[int]], ...] = ({None: {off}}, *idelta, {None: {0}})
    final = len(delta)-1
    for f in self.final:
      delta[f+off].setdefault(None, set()).add(final)
      for e in self.entry: delta[f+off][None].add(e+off)
    return NFA(delta, (0,), (final,))

  def plus(self) -> NFA: return self * self.star()

  def __mul__(self, other: NFA | Symbol) -> NFA: return self.concat(other)
  def __or__(self, other: NFA | Symbol) -> NFA: return self.union(other)
  def __pow__(self, p: int) -> NFA: return self.pow(p)

  def __repr__(self): return str(self)
  def __str__(self):
    return "\n".join((
      f"delta: {pformat(self.delta, indent=2, width=100)}",
      f"entry: {pformat(self.entry, indent=2, width=100)}",
      f"final: {pformat(self.final, indent=2, width=100)}"))

  def closure(self, states: set[int]) -> set[int]:
    c, p = states.copy(), states.copy()
    while len(p) > 0:
      a = set(self.delta[p.pop()].get(None, ()))
      c, p = c|a, p|a
    return c

  def move(self, states: set[int] | int, sym: Symbol) -> set[int]:
    return reduce(op.or_, (set(self.delta[s].get(sym, ())) for s in _cast(states, set)), set())

  def det(self) -> DFA:
    ini = _mask(*self.closure(set(self.entry)))
    fin = {ini} if ini & _mask(*self.final) else set()
    states = {ini}
    queue = {ini}
    vis = set()
    t = {}
    while len(queue) > 0:
      q = queue.pop()
      if q in vis: continue
      vis.add(q)
      for a in self.syms:
        dset = self.closure(self.move(_unmask(q), a))
        d = _mask(*dset)
        if d not in states:
          states.add(d)
          queue.add(d)
        t.setdefault(q, {})[a] = d
        if dset & self.final: fin.add(d)

    tbl = sorted(t.items(), key=lambda x: x[0])
    corr = {k: i for i, (k, _) in enumerate(tbl)}
    delta = [[0] * len(self.syms) for _ in range(len(tbl))]
    entry, final = corr[ini], {corr[x] for x in fin}
    for k, f in tbl:
      src = corr[k]
      for sym, dst in f.items():
        delta[src][self.syms.index(sym)] = corr[dst]

    return DFA(delta, entry, final, self.syms)

  def graphviz(self):
    g = Digraph(graph_attr={"rankdir":"LR"})
    g.node("entry", shape="point", width="0.1", height="0.1", label="")
    for i in range(len(self.delta)):
      attrs = {"label": str(i), "shape": "circle"}
      if i in self.final: attrs["shape"] = "doublecircle"
      g.node(str(i), **attrs)
      if i in self.entry: g.edge("entry", str(i))
    for i, state in enumerate(self.delta):
      for sym, transitions in state.items():
        for state_next in transitions:
          lbl = str(sym if sym is not None else "λ")
          g.edge(str(i), str(state_next), label=lbl)
    path = g.render(filename="graph", format="png")
    plt.imshow(mpimg.imread(path))
    plt.axis("off")
    plt.show(block=True)
    plt.close()

ab = NFA.symbols("ab")
cd = NFA.symbols("cd")

# abcd = ab * cd
# print(abcd.move(0, "b"))
# print(abcd)
# abcd.graphviz()
# abcd_det = abcd.det()
# abcd_det.graphviz()

# ab_cd = ab | cd
# ab_cd.det().graphviz()

# a3 = NFA.symbols("a").pow(3)
# a3.det().graphviz()

# TODO: the det() version of this does not work
astar = NFA.symbols("a").star()
astar.graphviz()
astar.det().graphviz()

# aplus = NFA.symbols("a").plus()
# aplus.graphviz()
# aplus.det().graphviz()
