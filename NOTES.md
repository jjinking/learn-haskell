### Sources
http://learnyouahaskell.com/

### General

- GHCi is the Glasgow Haskell Compiler interactive environment

- Change prompt
```haskell
:set prompt "ghci> "
```

- Enable multi-line
```haskell
:set +m
```

- Load script file in ghci
```haskell
ghci> :l fun.hs
```

- Enable multi-line
```haskell
ghci> :set +m
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

- When building up new lists from a list, build from the right, since `++` is much more expensie than `:`

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

- Type is determined by how many elements are inside, unlike lists; Functions for 2-tuples don't work for 3-tuples

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

### Types
- Check type of an expression:
```haskell
ghci> :t "a"
"a" :: [Char]
ghci> :t 'a'
'a' :: Char
```

- Types are usually written in capital case

- `Int` is bounded integer, `Integer` is unbounded

- `Float` is single-precision floating point, `Double` is double-precision

- There are infinite-many tuple types, since types take the size of tuple into account

## Generics

- Lowercase letters used to describe generic types
```haskell
ghci> :t head
head :: [a] -> a
ghci> :t snd
snd :: (a, b) -> b
```

## Typeclasses

- A **typeclass** is like 'interfaces' in Java

- The `Num` and `Eq` below are the class constraint
```haskell
ghci> :t (+)
(+) :: Num a => a -> a -> a
ghci> :t (==)
(==) :: Eq a => a -> a -> Bool
```

- `Eq`
  - Supports equality testing, requires members to implement `==` and `/=`
- `Ord`
  - Supports ordering, used in `<`, `>`, `<=` and `>=`
  - Requires membership in `Eq`
- `Show`
  - Supports print as string
- `Read`
  - Opposite of `Show`, can read string and evaluate
  - Can infer result type based on context, but w/o context, use **type annotations**
- `Num`
  - Numeric typeclass
  - Requires membership in `Show` and `Eq`


- Type annotations
```haskell
ghci> read "1" :: Float
1.0
```

- `fromIntegral` converts `Integral` to `Num` - useful function for dealing with numbers

### Patterns

- Define patterns in input to return the corresponding value, kind of like a bunch of if else

- Goes from top to bottom, so put most specific cases at the top
```haskell
threeOrFive :: (Integral a) => a -> String
threeOrFive 3 = "THREE"
threeOrFive 5 = "FIVE"
threeOrFive x = show (fromIntegral x :: Integer) ++ " IS NOT THREE OR FIVE!"
```

- Recursion with base case first
```haskell
factorial :: (Integral a) => a -> a
factorial 0 = 1
factorial n = n * factorial (n - 1)
```

- Grab the elements in input
```haskell
-- Adding two vectors in R2
addVectors :: (Num a) => (a, a) -> (a, a) -> (a, a)
addVectors a b = ((+) (fst a) (fst b), (+) (snd a) (snd b))
```
```haskell
-- Better way
addVectors2 :: (Num a) => (a, a) -> (a, a) -> (a, a)
addVectors2 (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)
```

- List comprehension
```haskell
ghci> let points = [(1,2), (3,4), (5,6)]
ghci> [(x + 1, y + 1) | (x, y) <- points]
[(2,3),(4,5),(6,7)]
```

- Use underscore to match and throw away unused elements, and use parentheses around the pattern when matching multiple values
```haskell
tell :: (Show a) => [a] -> String
tell [] = "The list is empty"
tell [x] = "The list has one element: " ++ show x
tell [x,y] = "The list has two elements: " ++ show x ++ " and " ++ show y
tell (x:y:_) = "This list has 3 or more elements, with first two elements "++ show x ++ " and " ++ show y
```

- Match entire input to use inside function
```haskell
matchall :: String -> String
matchall "" = "Empty string!"
matchall all@(x:xs) = "The first letter of " ++ all ++ " is " ++ [x]
```

### Guards

- Guards are kind of like `if else`
```haskell
ageTell :: (RealFloat a) => a -> String
ageTell age
  | age < 18.0 = "You're underage!"
  | age < 21.0 = "You're a young adult!"
  | age < 30.0 = "You're in your twenties!"
  | otherwise  = "You're getting older and wiser!"
```

- Defining functions **infix** with backticks
```haskell
myCompare :: (Ord a) => a -> a -> Ordering
a `myCompare` b
  | a > b     = GT
  | a == b    = EQ
  | otherwise = LT
```

- Also can use `where` and `let` to set names that can be used within function and comprehensions to set variables

### Case Expressions

```haskell
data Pet = Cat | Dog | Fish

hello :: Pet -> String
hello x = 
  case x of
    Cat -> "meeow"
    Dog -> "woof"
    Fish -> "bubble"
    
data Pet = Cat | Dog | Fish | Parrot String

hello :: Pet -> String
hello x = 
  case x of
    Cat -> "meeow"
    Dog -> "woof"
    Fish -> "bubble"
    Parrot name -> "pretty " ++ name
    _ -> "grunt"    

hello (Parrot "polly")
```

### Recursion
```haskell
replicate' :: (Num i, Ord i) => i -> a -> [a]
replicate' n x
  | n <= 0    = []
  | otherwise = x:replicate' (n-1) x
```

### Higher order functions take functions as parameters and returns functions

- Higher order functions: functions that take functions as parameter and outputs another function

- Haskell actually allows one parameter per function

- Curried functions - passing multiple values into a function is actually creating a function that takes in the first parameter, and then applying the second param, etc

- `map`, `filter`, `foldl`, `foldr`, `foldl1`, `foldl2`, `scanl`, `scanr` explained very well in this section

- Function application with `$` is right-associative, where as the regular space separators are left-associative. Also `$` can be used to use `map` to apply a list of functions to a single parameter value

- Function composition `f . g = \x -> f (g x)`
  - right-associative, makes it easy to create composed functions on the fly to pass to `map`
  - also useful to define functions in point free style
  - composing a long list of functions is bad style

### Modules

- `import x` to import module x

```haskell
import Data.List
import Data.List (nub, sort)
import Data.List hiding (nub)
import qualified Data.Map as Map
import qualified Data.Set as Set
```

- from ghci:

```bash
gchi> :m + Data.List Data.Map Data.Set Data.Char
```

### Custom types

- Think of the Haskell type/data system as 3 tiers:
  - Type class, i.e. Show, Num, which can be **subclassed**, which define behavior, similar to interfaces in Java
  - Value types, which are *instances* of a type class
  - Value, which are instances of a Value type.
- Use `class` keyword to create a custom type class
- Use `data` keyword followed by camelcase name of the type to create Value types
- `deriving` is used to show that the value type implements an interface (a type class)
  - If you use `deriving`, haskell will automatically generate an instance of `Show` for the new type.
- Instead of `deriving`, use `instance` keyword to make instances of typeclasses to customize behavior for the appropriate functions
  - Value types implement typeclass interfaces if every parameter (data constructors) in the type also implement the typeclass interface

``` haskell
-- Example of type constructors
data Point = Point Float Float deriving (Show)

-- "Type constructor" Shape has "data constructors" Circle and Rectangle, and Shape belongs to the "type class" Show
data Shape = Circle Point Float | Rectangle Point Point deriving (Show)

- Record syntax - automatically creates getters for the fields
data Person = Person {
	firstName :: String
	, lastName :: String
	, age :: Int
	, height :: Float
	, phoneNumber :: String
	, flavor :: String
	} deriving (Show)
```

- Type parameters - generics
  - `Nothing` is polymorphic, since it doesn't actually contain a value
  - Don't put *type* constraints into data declarations since function type params require them anyway
	  
- Type synonyms - create synonyms for already existing type using `type` keyword
  - convey more information about already-existing types, like person's name for string, etc

- Recursive data structures
  - Trees and Lists
  
  
Example of a **type class**
```haskell

-- Defining your own Show instance with customized behavior for `show` function
data Foo = Bar | Baz

instance Show Foo where
  show Bar = "it is a bar"
  show Baz = "this is a baz"


-- Any type `a` that belongs to the type class `Color` can use the functions `dark` and `lighten`
class Color a where
  dark :: a -> Bool
  lighten :: a -> a
      
data Bright = Blue | Red deriving (Read, Show)

darkBright :: Bright -> Bool
darkBright Blue = True
darkBright Red  = False

lightenBright :: Bright -> Bright
lightenBright Blue = Red
lightenBright Red = Red

-- An instance declaration says that a type is a member of a type class
instance Color Bright where
  dark = darkBright
  lighten = lightenBright
```

### **Functor** typeclass

Things that can be mapped over, like an *iterable*

### **Monad** typeclass

```haskell
 class Monad m where
        return :: a -> m a
        (>>=) :: m a -> (a -> m b) -> m b
        (>>)   :: m a -> m b -> m b
        fail   :: String -> m a
```

Example with `Maybe`

```haskell
data Maybe a = Just a | Nothing

import Control.Monad

instance Monad Maybe where
    return           =   Just
    Nothing  >>= f = Nothing
    (Just x) >>= f = f x
    fail _           =   Nothing
    
-- MonadPlus
instance MonadPlus Maybe where
    mzero               = Nothing
    Nothing `mplus` x = x
    x `mplus` _         = x
```

- Read learnyouahaskell
- Watch youtube video that i found that summarizes learnyouahaskell
- Watch the youtube video with fb employee
- Watch safarionline course



