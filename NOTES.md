### Sources
http://learnyouahaskell.com/

### General

- GHCi is the Glasgow Haskell Compiler interactive environment

- Change prompt
```haskell
:set prompt "ghci> "
```

- Load script file in ghci
```haskell
ghci> :l fun.hs
```

- Put parentheses around negative numbers
```haskell
3 * (-5)
```

- Boolean
```haskell
ghci> True && False
False
ghci> False || True
True
ghci> not False
True
ghci> not (True && True)
False
ghci> 2 == 2
True
ghci> 2 /= 2
True
ghci> "abc" == "abc"
True
```

- Strongly typed
  - `+` can only work between numeric types, i.e. ints and floats
  - `==` can only work between same types

- Numeric operators, i.e. `+` and `*` are *infix*
```haskell
ghci> div 92 10
9
ghci> 92 `div` 10
9
```

### Functions

- Most functions are *prefix* functions

- Haskell allows apostrophe in variable and function names

- Function names can't begin with uppercase letters

- Function f1 can be defined before f2, but f1 can use f2 inside its definition

### Control flow

- `if` statements in Haskell must be followed by `else` since it is an expression that must return a value

### Variables

- In GHCi, use `let` keyword to define a variable name
  - Don't have to do that in script file

### Lists

- Zero-indexed

- Homogenous - A list can only contain elements of the same type

- Strings (double quoted) are lists of chars (single quoted)
```haskell
"hello" == ['h','e','l','l','o']
```

- Concatenate lists using `++` operator, but not efficient for large lists
```haskell
ghci> [1,2,3] ++ [4]
[1,2,3,4]
```

- Inserting an element to beginning of list using `:` is fast
```haskell
ghci> 'H': "ello World"
"Hello World"
ghci> 1: [2,3]
[1,2,3]
ghci> 1:2:3:[]
[1,2,3]
```

- Access elements using `!!`
```haskell
ghci> "hello" !! 1
'e'
ghci> [1, 2, 3, 4, 5] !! 4
5
```

- Lists can hold different size lists of the same type only

- Comparison is based on ordered comparison of elements in each list
```haskell
ghci> [3, 2] > [1, 2, 3, 4, 5]
True
ghci> [3, 2] < [3, 2, 2]
True
```

- Theses functions work on lists with elements inside
```haskell
ghci> head [5,4,3,2,1]
5
ghci> tail [5,4,3,2,1]
[4,3,2,1]
ghci> last [5,4,3,2,1]
1
ghci> init [5,4,3,2,1]
[5,4,3,2]
```

- Other list functions
```haskell
ghci> length [5,4,3,2,1]
5
ghci> null [1,2,3]
False
ghci> null []
True
ghci> reverse [3,2,1]
[1,2,3]
ghci> take 3 [0,1,2,3,4]
[0,1,2]
ghci> drop 3 [0,1,2,3,4]
[3,4]
ghci> maximum [0,1,2]
2
ghci> minimum [0,1,2]
0
ghci> sum [0,1,2]
3
ghci> product [0,1,2]
0
ghci> elem 3 [0,1,2]
False
ghci> elem 2 [0,1,2]
True
ghci> 2 `elem` [0,1,2]
True
```

- Create finite lists with ranges
```haskell
ghci> [1..3] == [1,2,3]
True
ghci> ['a'..'Z']
""
ghci> ['A'..'z']
"ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz"
ghci> take 8 [2,4..]
[2,4,6,8,10,12,14,16]
ghci> take 10 (cycle [1,2,3])
[1,2,3,1,2,3,1,2,3,1]
ghci> take 11 (cycle "foo")
"foofoofoofo"
ghci> take 3 (repeat 1)
[1,1,1]
ghci> replicate 4 0
[0,0,0,0]
```

- List comprehension
```haskell
ghci> [x ** 2 | x <- [1..10]]
[1.0,4.0,9.0,16.0,25.0,36.0,49.0,64.0,81.0,100.0]
ghci> [x + 1 | x <- [1..10], (x `mod` 2) == 0]
[3,5,7,9,11]
```

- List comprehension as a function:
```haskell
ghci> let fooBars xs = [if (x `mod` 3) == 0 then "FOO" else "BAR!" | x <- xs, odd x]
ghci> fooBars [1..10]
["BAR!","FOO","BAR!","BAR!","FOO"]
```

- List comprehension with multiple predicates separated by commas
```haskell
ghci> [x | x <- [0..9], x /= 0, x /= 9]
[1,2,3,4,5,6,7,8]
```

- Draw from multiple lists
```haskell
ghci> [[x,y] | x <- [1,2,3], y <- [4,5,6]]
[[1,4],[1,5],[1,6],[2,4],[2,5],[2,6],[3,4],[3,5],[3,6]]
```

### Tuples

- Type is determined by how many elements are inside, unlike lists
  - Functions for 2-tuples don't work for 3-tuples

- A tuple doesn't necessarily contain a homogenous set of elements

- No singleton tuple since meaningless

- Uses parentheses

- Functions that work on pairs
```haskell
ghci> fst (0,1)
0
ghci> snd (0,"Zero")
"Zero"
```

- `zip` function creates a list of 2-tuples from 2 lists
```haskell
ghci> zip [0,1,2] ["Apple", "Orange"]
[(0,"Apple"),(1,"Orange")]
```

- Find right triangle with integer sides and hypotenuse less than or equal to 10, and perimeter 24
```haskell
ghci> [(a,b,c) | c <- [1..10], b <- [1..c], a <- [1..b], a^2 + b^2 == c^2, a+b+c == 24]
[(6,8,10)]
```
