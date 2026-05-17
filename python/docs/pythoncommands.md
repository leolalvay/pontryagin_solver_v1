# Quick Notes on Python 

## Class
- A class defines a blueprint for objects (instances) with state (attributes) and behavior (methods).
- `ClassName()` calls `__init__` to build a new instance.

## Constructor (`__init__`)
- The signature shows what you must pass when creating an instance.
- `self` is the instance being built; assign attributes to it.

```python
class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

p = Point(3, 4)  # p.x == 3, p.y == 4
```

## No-argument constructor
- If `__init__` only has `self`, instantiate with no args.

```python
class Counter:
    def __init__(self):
        self.value = 0

    def inc(self, step: int = 1):
        self.value += step

    def read(self) -> int:
        return self.value

c = Counter()
c.inc()
c.inc(3)
c.read()  # 4
```

## Methods and state
- Methods can mutate the instance; changes persist for later calls.
- Call order matters if one method depends on state set by another.

```python
class Processor:
    def __init__(self):
        self.data = None

    def load(self, x):
        self.data = x

    def run(self):
        if self.data is None:
            raise ValueError("Call load() first")
        return self.data ** 2

pr = Processor()
pr.load(5)
pr.run()  # 25
```

## Reset vs. new instance
- To get a “clean” object, create a new instance: `c2 = Counter()`.
- Or implement `reset()` to restore initial state.

```python
def reset(self):
    self.value = 0
```

## Key takeaways
- Constructor args = what you pass at `ClassName(...)`.
- `self.attr = ...` stores state; later methods read/modify it.
- Instances are independent: `Counter()` twice gives two separate states.
