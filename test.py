from __future__ import annotations
from typing import TypeAlias, Iterable
from pprint import pprint

Symbol: TypeAlias = str|None

class NFAl:
  def __init__(self,
               syms: tuple[Symbol,...] = (None,),
               delta: tuple[dict[Symbol, int],...] = ({None: 1}, {}),
               start: tuple[int,...] = (0,),
               final: tuple[int,...] = (1,)):
    nstate, nsym = len(delta), len(set(syms))
    assert nsym == len(syms), "duplicate entries in alphabet"
    assert all(0 <= s < nstate for s in start), "incorrect start state"
    assert all(0 <= s < nstate for s in final), "incorrect final state"
    self.nstate, self.nsym = nstate, nsym
    self.start, self.final = start, final
    self.delta, self.syms = delta, syms

  @staticmethod
  def single(s: Symbol) -> NFAl: return NFAl((s,), ({s: 1}, {}), (0,), (1,))
  @staticmethod
  def symbols(s: Iterable[Symbol]|Symbol) -> NFAl:
    syms = (None,) if s is None else tuple(s)
    delta = (*({syms[i]: i+1} for i in range(len(syms))), {})
    return NFAl(syms, delta, (0,), (len(delta)-1,))

  def concat(self, s: NFAl | Symbol) -> NFAl:
    o = s if isinstance(s, NFAl) else NFAl.single(s)
    syms = tuple(set(self.syms + o.syms))
    off = len(self.delta) # offset <-> second start

    # offset second NFAl states
    odelta = tuple({k: v+off for k, v in s.items()} for s in o.delta)
    delta = (*self.delta, *odelta)
    final = tuple(x+off for x in o.final)

    # bridge first with second NFAl
    for f in self.final: delta[f][None] = off

    return NFAl(syms, delta, self.start, final)

  def union(self, s: NFAl | Symbol) -> NFAl:
    o = s if isinstance(s, NFAl) else NFAl.single(s)
    syms = tuple(set(self.syms + o.syms))
    off1, off2 = 1, len(self.delta) + 1
    # delta = [{None: }]

    return self

  def __mul__(self, other: NFAl | Symbol) -> NFAl: return self.concat(other)
  def __or__(self, other: NFAl | Symbol) -> NFAl: return self.union(other)

  def print(self):
    pprint(f" syms: {self.syms}")
    pprint(f"delta: {self.delta}")
    pprint(f"start: {self.start}")
    pprint(f"final: {self.final}")

# l = NFAl()
# a = l.concat("a")
# ab = a.concat("b")
# ab.print()
ab = NFAl.symbols("ab")
cd = NFAl.symbols("cd")
abcd = ab * cd
abcd.print()
