from __future__ import annotations
import io
import operator as op
from pprint import pformat
from itertools import product
from collections import deque
from collections.abc import Iterable, Callable, Iterator
from functools import reduce
from types import MappingProxyType
from graphviz import Digraph
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

type Symbol = str | None

def _cast[T,C](v: T | Iterable[T], c: Callable[[Iterable[T]], C]) -> C: return c(v) if isinstance(v, Iterable) else c((v,))
def _mask(*v: int) -> int: return reduce(lambda a, b: a | (1 << b), v, 0)
def _mask_iter(m: int) -> Iterator[int]:
  while m:
    b = m & -m
    yield b.bit_length() - 1
    m ^= b
def _unmask(m: int) -> set[int]: return set(_mask_iter(m))

class StateTable:
  def __init__(self, n: int):
    self.n = n
    self.t: list[int|None] = [None] * int((n-1)*n/2)

  def _idx(self, key: tuple[int,int]) -> int:
    a, b = min(key[0], key[1]), max(key[0], key[1])
    assert a != b, "indeces cannot be the same"
    return (b-1)*b//2 + a

  def __getitem__(self, key: tuple[int,int]) -> int|None: return self.t[self._idx(key)]
  def __setitem__(self, key: tuple[int,int], value: int|None): self.t[self._idx(key)] = value

  def where(self, fn: Callable[[tuple[int,int],int|None], bool] | None) -> Iterator[tuple[int,int]]:
    return ((i, j) for i in range(self.n) for j in range(i) if fn is None or fn((i, j), self[i, j]))
  def unset(self) -> Iterator[tuple[int,int]]: return self.where(lambda _, v: v is None)
  def set(self) -> Iterator[tuple[int,int]]: return self.where(lambda _, v: v is not None)
  def indices(self) -> Iterator[tuple[int,int]]: return self.where(None)

  def __iter__(self): return iter(zip(self.indices(), self.t, strict=True))

  def __repr__(self) -> str:
    if self.n < 2: return "(no state pairs)"
    cells = ["·" if v is None else str(v) for v in self.t]     # · = unset pair
    w = max(len(c) for c in [*cells, f"s{self.n - 1}"])
    head = " " * w + " | " + " | ".join(f"s{j}".rjust(w) for j in range(self.n - 1))
    rows = (f"s{i}".rjust(w) + " | " + " | ".join(c.rjust(w) for c in cells[i*(i-1)//2 : i*(i+1)//2])
            for i in range(1, self.n))
    return "\n".join([head, *rows])

# TODO: Optimizations (ranked)
# 1. Predecessor propagation, O(n³k) → O(n²k): each sweep rescans all unset pairs; worst case O(n) sweeps. The classic fix: build reverse edges once (rev[t] = sources), seed a worklist with the final/nonfinal
#    pairs, and mark predecessor pairs as you pop — every pair marked exactly once, no sweeps. ~15 lines.
# 2. Hot-loop constants: every tbl[a, b] costs a tuple pack + min/max + assert + arithmetic; iterating enumerate(tbl.t) directly is 2-4x on the sweep.
# 3. Memory ceiling: n²/2 slots ≈ 400 MB at 10k states — that's the scale where Hopcroft's partition refinement (O(n) memory) would become necessary.
# 4. After fixing the annotation, store tuples in min()'s t like det() does — the two rebuild phases become the same idiom.
class DFA:
  __slots__ = ("_frozen", "_hash", "_min", "delta", "entry", "final", "syms")
  delta: tuple[tuple[int,...],...]
  entry: int
  final: frozenset[int]
  syms: tuple[str, ...]
  _hash: int
  _min: DFA | None

  def __init__(self, delta: tuple[tuple[int,...],...], entry: int, final: int | Iterable[int], syms: Iterable[Symbol]):
    object.__setattr__(self, "delta", tuple(map(tuple, delta)))
    object.__setattr__(self, "entry", int(entry))
    object.__setattr__(self, "final", _cast(final, frozenset))
    object.__setattr__(self, "syms", tuple(syms))
    object.__setattr__(self, "_min", None)
    object.__setattr__(self, "_hash", hash((self.delta, self.entry, self.final, self.syms)))
    object.__setattr__(self, "_frozen", True)

  def __setattr__(self, name, value):
    if getattr(self, "_frozen", False): raise AttributeError("DFA is immutable")
    object.__setattr__(self, name, value)

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, DFA): return NotImplemented
    return (self.delta, self.entry, self.final, self.syms) == (other.delta, other.entry, other.final, other.syms)
  def __hash__(self) -> int: return self._hash

  def __reduce__(self):  # bypass the freeze guard on unpickle: rebuild via constructor
    return (DFA, (self.delta, self.entry, self.final, self.syms))

  def min(self) -> DFA:
    if self._min is not None: return self._min
    m = self._minimize()
    object.__setattr__(self, "_min", m)
    return m

  def _minimize(self) -> DFA:
    n_states, n_syms = len(self.delta), len(self.syms)
    tbl = StateTable(n_states)
    nonfinal = [i for i in range(n_states) if i not in self.final]
    for a, b in product(self.final, nonfinal): tbl[a, b] = 0

    # table-filling Moore algorithm
    changes, i = True, 0
    while changes:
      changes = False
      i += 1
      for a, b in tbl.unset():
        for s in range(n_syms):
          da, db = self.delta[a][s], self.delta[b][s]
          if da != db and tbl[da, db] is not None:
            tbl[a, b] = i
            changes = True

    # find all state combinations to be merged and build a correlation (old id -> new id) dictionary
    # tbl.unset() is not enough as it only merges pairs of two at a time, requiring multiple min() calls
    merge = (v for i in range(tbl.n) if len(v := (i, *(j for j in range(i) if tbl[i, j] is None))) > 1)
    corr = {i: _mask(i) for i in range(n_states)}
    for v in merge:
      m = _mask(*v)
      for i in v: corr[i] = m

    # build transition dict (state id -> transitions) in bfs order
    # dicts preserve insertion order so entry = key 0
    t: dict[int,list[int]] = {}
    entry, final = corr[self.entry], {corr[f] for f in self.final}
    q = deque([entry])
    while q:
      s = q.popleft()
      if s in t: continue
      # all transitions *should* match for all merged states (only one transition is needed per state set)
      t[s] = [corr[c] for c in self.delta[next(_mask_iter(s))]]
      q += list(filter(lambda x: x not in t, t[s]))

    order = {s: i for i, s in enumerate(t)}
    delta = tuple(tuple(order[d] for d in row) for row in t.values())
    return DFA(delta, 0, {i for s, i in order.items() if s in final}, self.syms)

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

class NFA:
  __slots__ = ("_det", "_frozen", "_hash", "_key", "_syms", "delta", "entry", "final")
  delta: tuple[MappingProxyType[Symbol, frozenset[int]], ...]
  entry: frozenset[int]
  final: frozenset[int]
  _det: DFA | None
  _hash: int
  _key: tuple[tuple[Symbol, frozenset[int]], ...]
  _syms: tuple[str, ...] | None

  def __init__(self, delta: Iterable[dict[Symbol, set[int]]] = ({None: {1}}, {}), entry: int | Iterable[int] = 0, final: int | Iterable[int] = 1):
    d = tuple(MappingProxyType({k: frozenset(v) for k, v in tr.items()}) for tr in delta)
    entry, final = _cast(entry, frozenset), _cast(final, frozenset)
    assert all(0 <= s < len(d) for s in entry), f"incorrect entry states: 0 <= {entry} < {len(d)}"
    assert all(0 <= s < len(d) for s in final), f"incorrect final states: 0 <= {final} < {len(d)}"
    # canonical hashable form: sorted (symbol, targets) pairs, None (λ) first
    key = tuple(tuple(sorted(tr.items(), key=lambda kv: (kv[0] is not None, kv[0] or ""))) for tr in d)
    object.__setattr__(self, "delta", d)
    object.__setattr__(self, "entry", entry)
    object.__setattr__(self, "final", final)
    object.__setattr__(self, "_key", key)
    object.__setattr__(self, "_det", None)
    object.__setattr__(self, "_syms", None)
    object.__setattr__(self, "_hash", hash((key, entry, final)))
    object.__setattr__(self, "_frozen", True)

  def __setattr__(self, name, value):
    if getattr(self, "_frozen", False): raise AttributeError("NFA is immutable")
    object.__setattr__(self, name, value)

  def __eq__(self, other: object) -> bool:
    if not isinstance(other, NFA): return NotImplemented
    return (self._key, self.entry, self.final) == (other._key, other.entry, other.final)
  def __hash__(self) -> int: return self._hash

  def __reduce__(self):  # mappingproxy is not picklable: rebuild from plain dicts
    return (NFA, (tuple({k: set(v) for k, v in tr.items()} for tr in self.delta), self.entry, self.final))

  @property
  def syms(self) -> tuple[str, ...]:
    if self._syms is not None: return self._syms
    s = tuple(sorted({c for tr in self.delta for c in tr if c is not None}))
    object.__setattr__(self, "_syms", s)
    return s

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

  def det(self) -> DFA:
    if self._det is not None: return self._det
    d = self._determinize()
    object.__setattr__(self, "_det", d)
    return d

  def _determinize(self) -> DFA:
    entry, final = _mask(*self.closure(self.entry)), _mask(*self.final)

    n_syms = len(self.syms)
    # closed-move per (state, symbol): move and closure both distribute over union
    clos = [[_mask(*self.closure(self.move({s}, a))) for a in self.syms] for s in range(len(self.delta))]
    def c(s: int) -> tuple[int, ...]: return tuple(reduce(op.or_, (clos[i][a] for i in _mask_iter(s)), 0) for a in range(n_syms))

    # build transition dict (state id -> transitions) in bfs order
    # dicts preserve insertion order so entry = key 0
    t: dict[int,tuple[int,...]] = {entry: ()}
    q = deque([entry])
    while q:
      s = q.popleft()
      row = t[s] = c(s)
      for d in row:
        if d not in t:
          t[d] = ()
          q.append(d)

    order = {s: i for i, s in enumerate(t)}
    delta = tuple(tuple(order[d] for d in row) for row in t.values())
    return DFA(delta, 0, {i for s, i in order.items() if s & final}, self.syms)

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

def same_language(a: NFA, b: NFA) -> bool:
  return a.det().min() == b.det().min()

def show(*graphs: Digraph, cols: int = 2, block: bool = False):
  imgs = [mpimg.imread(io.BytesIO(g.pipe(format="png")), format="png") for g in graphs]
  n = len(imgs)
  rows = (n + cols - 1) // cols

  _, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows), squeeze=False)
  for ax, im in zip(axes.flat, imgs, strict=False):
    ax.imshow(im)
    ax.axis("off")
  for ax in axes.flat[n:]: ax.axis("off")

  plt.tight_layout()
  plt.show(block=block)   # no plt.close: non-blocking windows must stay alive

if __name__ == "__main__":
  a = NFA.symbols("a")
  b = NFA.symbols("b")
  a_b = (a | b).star()
  show(a_b.graphviz(), a_b.det().graphviz(), a_b.det().min().graphviz())

  ab = NFA.symbols("ab")
  cd = NFA.symbols("cd")

  abcd = ab * cd
  show(abcd.graphviz(), abcd.det().graphviz(), abcd.det().min().graphviz())

  ab_cd = (ab | cd).star()
  show(ab_cd.graphviz(), ab_cd.det().graphviz(), ab_cd.det().min().graphviz())

  a3 = NFA.symbols("a").pow(3)
  show(a3.graphviz(), a3.det().graphviz(), a3.det().min().graphviz())

  astar = NFA.symbols("a").star()
  show(astar.graphviz(), astar.det().graphviz(), astar.det().min().graphviz())

  aplus = NFA.symbols("a").plus()
  show(aplus.graphviz(), aplus.det().graphviz(), aplus.det().min().graphviz())

  plt.show()   # block once until all windows are closed
