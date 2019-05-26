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
:l fun.hs
-- reload current script
:r
```

- Enable multi-line
```haskell
:set +m
```

- Numeric operators, i.e. `+` and `*` are *infix*
```haskell
-- prefix function
ghci> div 92 10
9
-- infix form of same function
ghci> 92 `div` 10
9
```

### Functions

- Haskell allows apostrophe in variable and function names
  - Usually use ' to either denote a strict version of a function (one that isn't lazy) or a slightly modified version of a function or a variable

- Function names can't begin with uppercase letters

- Function f1 can be defined before f2, but f1 can use f2 inside its definition

### Control flow

- `if` statements in Haskell must be followed by `else` since it is an expression that must return a value

### Lists

- Zero-indexed (but no random access in const time)

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


- Inserting an element to beginning of list using `cons` (or operator `:`) is fast
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

- Comparison is based on ordered comparison of elements in each list
```haskell
ghci> [3, 2] > [1, 2, 3, 4, 5]
True
ghci> [3, 2] < [3, 2, 2]
True
```

- List functions
```haskell
-- head, tail, last, init not safe with empty lists
ghci> head [5,4,3,2,1]
5
ghci> head []  
*** Exception: Prelude.head: empty list
ghci> tail [5,4,3,2,1]
[4,3,2,1]
ghci> last [5,4,3,2,1]
1
ghci> init [5,4,3,2,1]
[5,4,3,2]

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

-- List comprehension as a function:
ghci> let fooBars xs = [if (x `mod` 3) == 0 then "FOO" else "BAR!" | x <- xs, odd x]
ghci> fooBars [1..10]
["BAR!","FOO","BAR!","BAR!","FOO"]

-- List comprehension with multiple predicates separated by commas
ghci> [x | x <- [0..9], x /= 0, x /= 9]
[1,2,3,4,5,6,7,8]

-- Draw from multiple lists
ghci> [[x,y] | x <- [1,2,3], y <- [4,5,6]]
[[1,4],[1,5],[1,6],[2,4],[2,5],[2,6],[3,4],[3,5],[3,6]]

-- Nested lists
ghci> let xxs = [[1,3,5,2,3,1,2,4,5],[1,2,3,4,5,6,7,8,9],[1,2,4,2,1,6,3,1,3,2,3,6]]  
ghci> [ [ x | x <- xs, even x ] | xs <- xxs]  
[[2,2,4],[2,4,6,8],[2,4,2,6,2,6]]
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

-- `zip` function creates a list of 2-tuples from 2 lists
ghci> zip [0..] ["Apple", "Orange"]
[(0,"Apple"),(1,"Orange")]
```

- Find right triangle with integer sides and hypotenuse less than or equal to 10, and perimeter 24
```haskell
ghci> [(a,b,c) | c <- [1..10], b <- [1..c], a <- [1..b], a^2 + b^2 == c^2, a+b+c == 24]
[(6,8,10)]
```

### Types

- Types are usually written in capital case
- `Int` is bounded integer, `Integer` is unbounded but inefficient

## Generics

- Lowercase letters usually one of {a, b, c, d} used to describe generic types
```haskell
ghci> :t head
head :: [a] -> a
ghci> :t snd
snd :: (a, b) -> b
```

## Typeclasses

- A **typeclass** is like `interface` in Java
- Set **class constraints** using `=>`

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
  - Member types also implement `compare`
  - Requires membership in `Eq`
- `Show`
  - Supports print as string
  - Member types implement `show`
- `Read`
  - Member types implement `read`
  - Opposite of `Show`, can read string and evaluate
  - Can infer result type based on context, but w/o context, use **type annotations** ex: `read "5" :: Int`
- `Enum`
  - Member types implement `succ` and `pred` functions
- `Bounded`
  - Member types implement `minBound` and `maxBound` which are "polymorphic constants"
  - All tuples whose components are members of `Bounded` are also in `Bounded`
- `Num`
  - Numeric typeclass
  - Requires membership in `Show` and `Eq`
 
```haskell
-- Integer-number {..-1, -1, 0, 1, 2} literals are also "polymorphic constants"
ghci> :t 20
20 :: (Num t) => t  
ghci> 20 :: Int  
20  
ghci> 20 :: Integer  
20  
ghci> 20 :: Float  
20.0  
ghci> 20 :: Double  
20.0  
```

- `Integral` includes `Int` and `Integer`
  - `fromIntegral` converts `Integral` to `Num` - useful function for dealing with numbers, e.g. ` fromIntegral (length [1,2,3,4]) + 3.2`
- `Floating` includes `Float` and `Double`


### Patterns

- Goes from top to bottom, so put most specific cases at the top (base case)
```haskell
threeOrFive :: (Integral a) => a -> String
threeOrFive 3 = "THREE"
threeOrFive 5 = "FIVE"
threeOrFive x = show (fromIntegral x :: Integer) ++ " IS NOT THREE OR FIVE!"

-- Recursion with base case first
factorial :: (Integral a) => a -> a
factorial 0 = 1
factorial n = n * factorial (n - 1)
```

- Grab the elements in input **tuples**
```haskell
-- Adding two vectors in R2 by grabbing their individual components
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

- Match entire input to use inside function using `@`, or *as patterns*
```haskell
matchall :: String -> String
matchall "" = "Empty string!"
matchall all@(x:xs) = "The first letter of " ++ all ++ " is " ++ [x]
```

### Guards

- Guards are similar to `if else`

```haskell
ageTell :: (RealFloat a) => a -> String
ageTell age
  | age < 18.0 = "You're underage!"
  | age < 21.0 = "You're a young adult!"
  | age < 30.0 = "You're in your twenties!"
  | otherwise  = "You're getting older and wiser!"

-- Defining functions **infix** with backticks
myCompare :: (Ord a) => a -> a -> Ordering
a `myCompare` b
  | a > b     = GT
  | a == b    = EQ
  | otherwise = LT

-- Using `where` for the DRY principle
bmiTell :: (RealFloat a) => a -> a -> String
bmiTell weight height
    | bmi <= skinny = "You're underweight, you emo, you!"
    | bmi <= normal = "You're supposedly normal. Pffft, I bet you're ugly!"
    | bmi <= fat    = "You're fat! Lose some weight, fatty!"
    | otherwise     = "You're a whale, congratulations!"
    where bmi = weight / height ^ 2
          skinny = 18.5
          normal = 25.0
          fat = 30.0

-- Pattern match inside `where` bindings
initials :: String -> String -> String
initials firstname lastname = [f] ++ ". " ++ [l] ++ "."
    where (f:_) = firstname
          (l:_) = lastname
	  
-- Define function `bmi` in `where` block
calcBmis :: (RealFloat a) => [(a, a)] -> [a]
calcBmis xs = [bmi w h | (w, h) <- xs]
    where bmi weight height = weight / height ^ 2
```

### Let

- `where` bindings are syntactic constructs
- `let` bindings are expressions that evaluate a value, e.g. `4 * (let a = 9 in a + 1) + 2 `
- `let` bindings are similar to `where` bindings, but more local; it doesn't span across guards.
- form: `let <bindings> in <expression>`

```haskell
cylinder :: (RealFloat a) => a -> a -> a
cylinder r h =
    let sideArea = 2 * pi * r * h
        topArea = pi * r ^2
    in  sideArea + 2 * topArea
    
 -- Binding several variables inline
 (let a = 100; b = 200; c = 300 in a*b*c, let foo="Hey "; bar = "there!" in foo ++ bar)
(6000000,"Hey there!")

-- List comprehension
calcBmis :: (RealFloat a) => [(a, a)] -> [a]
calcBmis xs = [bmi | (w, h) <- xs, let bmi = w / h ^ 2]
```

### Case Expressions

- Pattern matching on parameters in function definitions is syntactic sugar for using case expressions

```haskell
head' :: [a] -> a
head' [] = error "No head for empty lists!"
head' (x:_) = x

-- Same as the following
head' :: [a] -> a
head' xs = case xs of [] -> error "No head for empty lists!"
                      (x:_) -> x

-- Case expressions are not limited to function definitions
describeList :: [a] -> String
describeList xs = "The list is " ++ case xs of [] -> "empty."
                                               [x] -> "a singleton list."
                                               xs -> "a longer list."

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
-- `Num` is not subclass of `Ord`, so we have to specify both for subtraction and comparison
replicate' :: (Num i, Ord i) => i -> a -> [a]
replicate' n x
  | n <= 0    = []
  | otherwise = x:replicate' (n-1) x
```

### Higher order functions take functions as parameters and returns functions

- Higher order functions are functions that take functions as parameter and/or outputs another function

- Haskell actually allows one parameter per function via currying
  - Currying functions - converting a function that takes arg1..argN into a function that takes arg1, and returns a function that takes arg2, which returns a function that takes arg3, and so on.

- `map`, `filter`, `foldl`, `foldr`, `foldl1`, `foldl2`, `scanl`, `scanr` explained in detail [here](http://learnyouahaskell.com/higher-order-functions)

- Usually use `foldr` when building up new lists from an input list

- You cannot define several pattern-matching for one parameter when defining **lambdas**

- Function application with `$` is right-associative, where as the regular space separators are left-associative

```haskell
sum (map sqrt [1..130])
sum $ map sqrt [1..130]

sqrt (3 + 4 + 9)
sqrt $ 3 + 4 + 9

f (g (z x))
f $ g $ z x

sum (filter (> 10) (map (*2) [2..10]))
sum $ filter (> 10) $ map (*2) [2..10]

-- `$` can be used to use `map` to apply a list of functions to a single parameter value
ghci> map ($ 3) [(4+), (10*), (^2), sqrt]
[7.0,30.0,9.0,1.7320508075688772]
```

- Function composition `f . g = \x -> f (g x)`
  - composed functions are right-associative
  
```haskell
replicate 100 (product (map (*3) (zipWith max [1,2,3,4,5] [4,5,6,7,8])))
replicate 100 . product . map (*3) . zipWith max [1,2,3,4,5] $ [4,5,6,7,8]

-- Function composition is useful to define functions in *point free* style (removing the `x` arg from both sides of `=` sign)
fn x = ceiling (negate (tan (cos (max 50 x))))
fn = ceiling . negate . tan . cos . max 50

-- Composing a long list of functions is bad style
oddSquareSum :: Integer  
oddSquareSum = sum . takeWhile (<10000) . filter odd . map (^2) $ [1..]  
-- This is more readable
oddSquareSum :: Integer  
oddSquareSum =   
    let oddSquares = filter odd $ map (^2) [1..]  
        belowLimit = takeWhile (<10000) oddSquares  
    in  sum belowLimit  
```

### Modules

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

[haskell Standard Library modules list](https://downloads.haskell.org/~ghc/latest/docs/html/libraries/)

Watch out for lazy folds. They produce **thunks**, a promise that a function will compute its value when asked to actually produce the result. May cause stack overflow.

`concatMap` is like flatmap
```haskell
ghci> concatMap (replicate 4) [1..3]
[1,1,1,1,2,2,2,2,3,3,3,3]
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
- Read articles
  - https://www.lambdacat.com/the-midnight-monad-a-journey-to-enlightenment/
  - https://en.wikibooks.org/wiki/Haskell/Understanding_monads
  - Monads are burritos
    - https://blog.plover.com/prog/burritos.html
    - https://chrisdone.com/posts/monads-are-burritos/
    - https://byorgey.wordpress.com/2009/01/12/abstraction-intuition-and-the-monad-tutorial-fallacy/
  - http://blog.sigfpe.com/2006/08/you-could-have-invented-monads-and.html
  - http://web.archive.org/web/20081206204420/http://www.loria.fr/~kow/monads/index.html

