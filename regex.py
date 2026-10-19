from __future__ import annotations
import io
import operator as op
from pprint import pformat
from collections import deque
from collections.abc import Iterable, Callable
from functools import reduce, cached_property
from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

type Symbol = str | None

def _cast[T,C](v: T | Iterable[T], c: Callable[[Iterable[T]], C]) -> C: return c(v) if isinstance(v, Iterable) else c((v,))
def _mask(*v: int) -> int: return reduce(lambda a, b: a | (1 << b), v, 0)
def _mask_iter(m: int) -> Iterable[int]:
  while m:
    b = m & -m
    yield b.bit_length() - 1
    m ^= b
def _unmask(m: int) -> set[int]: return set(_mask_iter(m))

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

  def graphviz(self) -> Digraph:
    g = Digraph(graph_attr={"rankdir": "LR", "dpi": "300"})
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
    return g

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
    delta: tuple[dict[Symbol, set[int]], ...] = ({None: {off, len(idelta)+1}}, *idelta, {None: set()})
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

  def closure(self, states: Iterable[int] | int) -> set[int]:
    c, p = _cast(states, set), _cast(states, set)
    while len(p) > 0:
      a = set(self.delta[p.pop()].get(None, ())) - c
      c, p = c|a, p|a
    return c

  def move(self, states: Iterable[int] | int, sym: Symbol) -> set[int]:
    return reduce(op.or_, (set(self.delta[s].get(sym, ())) for s in _cast(states, set)), set())

  def det(self):
    entry, final = _mask(*self.closure(self.entry)), _mask(*self.final)

    n_syms = len(self.syms)
    # closed-move per (state, symbol): move and closure both distribute over union
    clos = [[_mask(*self.closure(self.move({s}, a))) for a in self.syms] for s in range(len(self.delta))]
    def c(s: int) -> tuple[int, ...]: return tuple(reduce(op.or_, (clos[i][a] for i in _mask_iter(s)), 0) for a in range(n_syms))

    # mark-on-enqueue worklist: every state enters q exactly once (no dedup guard
    # needed), and t's insertion order IS the BFS discovery order
    t: dict[int, tuple[int, ...]] = {entry: ()}
    q = deque([entry])
    while q:
      s = q.popleft()
      row = t[s] = c(s)
      for d in row:
        if d not in t:
          t[d] = ()
          q.append(d)

    corr = {s: i for i, s in enumerate(t)} # BFS numbering for free (dicts preserver insertion order); entry == key 0
    delta = [[corr[d] for d in row] for row in t.values()]
    return DFA(delta, 0, {i for s, i in corr.items() if s & final}, self.syms)

  def graphviz(self) -> Digraph:
    g = Digraph(graph_attr={"rankdir": "LR", "dpi": "300"})
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
    return g

def show(*graphs: Digraph, cols: int = 2):
  imgs = [mpimg.imread(io.BytesIO(g.pipe(format="png")), format="png") for g in graphs]
  n = len(imgs)
  rows = (n + cols - 1) // cols

  fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
  for ax, im in zip(axes.flat, imgs, strict=True):
    ax.imshow(im)
    ax.axis("off")
  for ax in axes.flat[n:]: ax.axis("off")

  plt.tight_layout()
  plt.show(block=True)
  plt.close(fig)

ab = NFA.symbols("ab")
cd = NFA.symbols("cd")

# abcd = ab * cd
# print(abcd.move(0, "b"))
# print(abcd)
# abcd.graphviz()
# abcd_det = abcd.det()
# abcd_det.graphviz()

# ab_cd = (ab | cd).star()
# show(ab_cd.graphviz(), ab_cd.det().graphviz())

# a3 = NFA.symbols("a").pow(3)
# a3.det().graphviz()

astar = NFA.symbols("a").star()
show(astar.graphviz(), astar.det().graphviz())

# aplus = NFA.symbols("a").plus()
# aplus.graphviz()
# aplus.det().graphviz()
