### Sources
http://learnyouahaskell.com/

## General

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

## Recursion

```haskell
-- `Num` is not subclass of `Ord`, so we have to specify both for subtraction and comparison
replicate' :: (Num i, Ord i) => i -> a -> [a]
replicate' n x
  | n <= 0    = []
  | otherwise = x:replicate' (n-1) x
```

## Higher order functions take functions as parameters and returns functions

- Higher order functions are functions that take functions as parameter and/or outputs another function

- Haskell actually allows one parameter per function via currying
  - **Currying** functions - converting a function that takes arg1..argN into a function that takes arg1, and returns a function that takes arg2, which returns a function that takes arg3, and so on.
  - Currying enables us to create **partial** functions, which are functions that have some of their parameters already applied to them.

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

## Modules

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

## Custom types

Think of the Haskell type/data system as 3 tiers:
  1. Type class, i.e. Show, Num, which can be **subclassed**, which define behavior, similar to interfaces in Java
  2. Value types (ADT, or Algebraic Data Types), which are *instances* of a *type class*
  3. Value, which are instances of a *value type*
  
- Use `class` keyword to create a custom type class
- Use `data` keyword followed by camelcase name of the type to create *value types* (ADT)
- `deriving` is used to show that the value type implements an interface (a type class)
  - If you use `deriving`, haskell will automatically generate an instance of `Show` for the new type.
- Instead of `deriving`, use `instance` keyword to make instances of typeclasses to customize behavior for the appropriate functions
  - Value types implement typeclass interfaces if every parameter (data|value constructors) in the type also implement the typeclass interface

``` haskell
-- Example of custom types
data Bool = False | True
data Int = -2147483648 | -2147483647 | ... | -1 | 0 | 1 | 2 | ... | 2147483647
data Point = Point Float Float deriving (Show)

-- Value type `Shape` has "data|value constructors" `Circle` and `Rectangle`
-- `Shape` belongs to the "type class" Show
-- `Circle` and `Rectangle` are functions
data Shape = Circle Point Float | Rectangle Point Point deriving (Show)

-- Pattern match against value constructors
surface :: Shape -> Float  
surface (Circle _ r) = pi * r ^ 2  
surface (Rectangle (Point x1 y1) (Point x2 y2)) = (abs $ x2 - x1) * (abs $ y2 - y1)
```

### Exporting

Don't export value constructors to hide implementation details, and force users to use certain functions to create instance of types. Also, users won't be able to pattern match against the value constructors

```haskell
module Shapes   
( Point(..)  
, Shape(..)  -- same as `Shape (Rectangle, Circle)`
, surface  
, nudge  
, baseCircle  
, baseRect  
) where
```

### Record syntax
automatically creates getters for the fields

```haskell
-- Cumbersome to write getter methods for all the fields
data Person = Person String String Int Float String String deriving (Show)

-- Better
data Person = Person {
	firstName :: String,
	lastName :: String,
	age :: Int,
	height :: Float,
	phoneNumber :: String,
	flavor :: String
	} deriving (Show)
```

### Type Constructors Use Type Parameters

Type constructors take types as parameters to produce concrete types, giving us *generics* similar to Java and C++

```haskell
-- Maybe **not a type**. It is a **type constructor** since it takes a type parameter
-- It takes a "type argument" `a` to "construct a type", so it's a "type constructor"
-- `Nothing` is polymorphic, since it doesn't actually contain a value
data Maybe a = Nothing | Just a

-- Don't put *type* constraints into data declarations since function type params require them anyway
data (Ord k) => Map k v = ...  
```

### Type synonyms

Create synonyms for already existing type using `type` keyword to convey more information about already-existing types, like person's name for string, etc

```haskell
type String = [Char]
type PhoneNumber = String
type Name = String
type PhoneBook = [(String,String)]
type PhoneBook = [(Name,PhoneNumber)]

-- Parameterized type synonyms are type constructors
type AssocList k v = [(k,v)]

-- Partially apply type parameters to get new type constructors
type IntMap v = Map.Map Int v
type IntMap = Map.Map Int
```

### Recursive data structures

```haskell
data List a = Empty | Cons a (List a) deriving (Show, Read, Eq, Ord)

data Tree a = EmptyTree | Node a (Tree a) (Tree a) deriving (Show, Read, Eq)

singleton :: a -> Tree a
singleton x = Node x EmptyTree EmptyTree

treeInsert :: (Ord a) => a -> Tree a -> Tree a
treeInsert x EmptyTree = singleton x
treeInsert x (Node a left right)
    | x == a = Node x left right
    | x < a  = Node a (treeInsert x left) right
    | x > a  = Node a left (treeInsert x right)
    
treeElem :: (Ord a) => a -> Tree a -> Bool
treeElem x EmptyTree = False
treeElem x (Node a left right)
    | x == a = True
    | x < a  = treeElem x left
    | x > a  = treeElem x right
    
-- Create a BST from a list of numbers
let numsTree = foldr treeInsert EmptyTree nums
```

## Typeclasses

```haskell
-- `Eq` typeclass in standard prelude
-- Here `a` represents a type that is an instance of `Eq`
-- The recursive definitions of `==` and `/=` enable easy "minimal complete definition" for instances
class Eq a where
    (==) :: a -> a -> Bool
    (/=) :: a -> a -> Bool
    x == y = not (x /= y)
    x /= y = not (x == y)
    
-- Example instance
data TrafficLight = Red | Yellow | Green

-- Just need to implement `==` since `\=` is defined in the typeclass definition
instance Eq TrafficLight where
    Red == Red = True
    Green == Green = True
    Yellow == Yellow = True
    _ == _ = False
    
-- Manually defining instance of `Show` instead of using `deriving Show`
-- Deriving would have just translated value constructors to strings, but here we can add "light"
instance Show TrafficLight where
    show Red = "Red light"
    show Yellow = "Yellow light"
    show Green = "Green light"

-- Another example

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

### Typeclasses that extend (subclass) other typeclasses

Implemented using class constraints in *class* declarations

```haskell
--- `Num` typeclass requires that a be an instance of `Eq` also
class (Eq a) => Num a where
	...
```

### Type constructors as instances of typeclasses

```haskell
-- `Maybe` by itself is not a concrete type since it is a type constructor, so we use a variable `m`
-- since in the class definition above, the `a` represents a concrete type
-- Also, m has to be in `Eq` since we are using `==` on x and y, which are instances of m
instance (Eq m) => Eq (Maybe m) where
  Just x == Just y = x == y
  Nothing == Nothing = True
  _ == _ = False
  
-- Get info about a typeclass and all types that are instances of it
:info Show

-- Get info about a type or type constructor, and all typeclasses that the type is in
:info Maybe

-- Get info about a function and its type signature
:info map
```

### *Functor* typeclass

```haskell
-- Looking at the `fmap` type annotation, `f` represents a type constructor, not a concrete type
class Functor f where
    fmap :: (a -> b) -> f a -> f b
    
-- Use a type constructor, i.e. `[]` since `f` above represents a type constructor
instance Functor [] where
    fmap = map
    
-- Simple example
instance Functor Maybe where
    fmap f (Just x) = Just (f x)
    fmap f Nothing = Nothing
    
-- Recursive example
instance Functor Tree where
    fmap f EmptyTree = EmptyTree
    fmap f (Node x leftsub rightsub) = Node (f x) (fmap f leftsub) (fmap f rightsub)
    
-- The `Functor` typeclass wants a type constructor that takes only one type parameter
-- Partially apply the left parameter so there's only one free parameter
-- `Either a` is a type constructor with one type parameter
instance Functor (Either a) where
    fmap f (Right x) = Right (f x)
    fmap f (Left x) = Left x
```

IO Functor example

```haskell
import Data.Char
import Data.List

main = do line <- fmap (intersperse '-' . reverse . map toUpper) getLine
          putStrLn line
```

`(->) r` is a type constructor with a single concrete type parameter, and a functor. Using `fmap` on functions is function composition. Note: (-> r a) is the same as r -> a

`fmap :: (a -> b) -> (f a -> f b)` shows that `fmap` *lifts* a function `a -> b` to `f a -> f b`

#### Laws

identity `id` and function composition `f.g`

### Applicatives

Typeclass that defines two methods, `pure` and `<*>`

```haskell
-- Applicatives require members to be Functors
class (Functor f) => Applicative f where
    pure :: a -> f a
    (<*>) :: f (a -> b) -> f a -> f b
    
-- Applicative instance for Maybe
instance Applicative Maybe where
    pure = Just
    Nothing <*> _ = Nothing
    (Just f) <*> something = fmap f something

ghci> Just (+3) <*> Just 9
Just 12
ghci> pure (+3) <*> Just 10
Just 13
ghci> pure (+3) <*> Just 9
Just 12
ghci> Just (++"hahah") <*> Nothing
Nothing
ghci> Nothing <*> Just "woot"
Nothing

ghci> pure (+) <*> Just 3 <*> Just 5
Just 8
ghci> pure (+) <*> Just 3 <*> Nothing
Nothing
ghci> pure (+) <*> Nothing <*> Just 5
Nothing
```


### *Kinds*

kind = type of a type

```haskell
-- * means Int is a "concrete type", which doesn't take any type params (btw, functions are also concrete)
ghci> :k Int
Int :: *

-- Maybe type constructor takes a concrete type (ex: Int) and returns a concrete type (ex: Maybe Int)
ghci> :k Maybe
Maybe :: * -> *
ghci> :k Maybe Int
Maybe Int :: *

-- Type constructors are curried, so we can partially apply them.
ghci> :k Either
Either :: * -> * -> *
ghci> :k Either String
Either String :: * -> *
ghci> :k Either String Int
Either String Int :: *
```

### Type-foo

Perhaps over-indulging in "type theory" aka "type theory porn"

```haskell
-- t has to have kind * -> (* -> *) -> *
class Tofu t where
    tofu :: j a -> t a j

-- Creating a type with a kind like 't' above
data Frank a j  = Frank {frankField :: j a} deriving (Show)

-- Creating some Frank values
ghci> :t Frank {frankField = Just "HAHA"}  
Frank {frankField = Just "HAHA"} :: Frank [Char] Maybe  
ghci> :t Frank {frankField = Node 'a' EmptyTree EmptyTree}  
Frank {frankField = Node 'a' EmptyTree EmptyTree} :: Frank Char Tree  
ghci> :t Frank {frankField = "YES"}  
Frank {frankField = "YES"} :: Frank Char []  

-- Making Frank an instance of tofu
instance Tofu Frank where
    tofu x = Frank x
    
ghci> tofu (Just 'a') :: Frank Char Maybe
Frank {frankField = Just 'a'}
ghci> tofu ["HELLO"] :: Frank [Char] []
Frank {frankField = ["HELLO"]}
```

## I/O

A way to deal with side-effects

```haskell
-- putStrLn takes a String and returns I/O action that has a result type of (), or unit
-- The empty tuple is a value of () and it also has a type of ()
ghci> :t putStrLn
putStrLn :: String -> IO ()
ghci> :t putStrLn "hello, world"
putStrLn "hello, world" :: IO ()
```

I/O action only runs (e.g. prints to screen, or reads from input) inside `main`. Otherwise, they are just values that don't do anything.

```haskell
-- `main` always has a type signature of `main :: IO a`
-- Convention is to not specify a type declaration for main
main = do
    putStrLn "Hello, what's your name?"
    name <- getLine
    putStrLn ("Hey " ++ name ++ ", you rock!")
```

Using `let` bindings inside `do` block

```haskell
import Data.Char  

main = do  
    putStrLn "What's your first name?"
    firstName <- getLine
    putStrLn "What's your last name?"
    lastName <- getLine
    let bigFirstName = map toUpper firstName
        bigLastName = map toUpper lastName
    putStrLn $ "hey " ++ bigFirstName ++ " " ++ bigLastName ++ ", how are you?"
```

Using `return` and conditionals

```haskell
main = do
    line <- getLine
    if null line
        then return ()
        else do
            putStrLn $ reverseWords line
            main
  
reverseWords :: String -> String
reverseWords = unwords . map reverse . words
```

`return` makes an I/O action out of a pure value. It **does not** end execution of a function. It's the opposite of `<-`

```haskell
main = do
    a <- return "hell"
    b <- return "yeah!"
    putStrLn $ a ++ " " ++ b

main = do
    let a = "hell"
        b = "yeah"
    putStrLn $ a ++ " " ++ b
```

Recursion in I/O

```haskell
putStr :: String -> IO ()
putStr [] = return ()
putStr (x:xs) = do
    putChar x
    putStr xs
```

Print just converts a value to `String` by calling `show` on it, then writes to terminal using `putStrLn`. GHCI uses `print` in the print stage of the repl.

```haskell
print = putStrLn . show

main = do   print True
            print 2
            print "haha"
            print 3.2
            print [3,4,3]
```

### Files and Streams

`getContents` reads from std input line by line until end-of-file, or ctrl-d

```haskell
import Data.Char

main = do
    contents <- getContents
    putStr (map toUpper contents)
```

`interact` implements a common pattern of taking a string from input, transforming it, then outputting

```haskell
main = interact shortLinesOnly  

shortLinesOnly :: String -> String
shortLinesOnly input =
    let allLines = lines input
        shortLines = filter (\line -> length line < 10) allLines
        result = unlines shortLines
    in  result 
```

Reading a file and printing to screen

```haskell
import System.IO

-- openFile :: FilePath -> IOMode -> IO Handle
-- type FilePath = String
-- data IOMode = ReadMode | WriteMode | AppendMode | ReadWriteMode
-- hGetContents is like getContents, but from a file handle
main = do
    handle <- openFile "girlfriend.txt" ReadMode
    contents <- hGetContents handle
    putStr contents
    hClose handle
    
-- Using `withfile` no need to close file after
import System.IO

main = do
    withFile "girlfriend.txt" ReadMode (\handle -> do
        contents <- hGetContents handle
        putStr contents)
	
-- Reading contents from file with `readFile` is shorter
import System.IO

main = do
    contents <- readFile "girlfriend.txt"
    putStr contents
    
-- Reading from a file, then writing to another file
import System.IO
import Data.Char

main = do
    contents <- readFile "girlfriend.txt"
    writeFile "girlfriendcaps.txt" (map toUpper contents)
```

### Command Line Args

```haskell
import System.Environment
import Data.List

main = do
   args <- getArgs
   progName <- getProgName
   putStrLn "The arguments are:"
   mapM putStrLn args
   putStrLn "The program name is:"
   putStrLn progName
```

```bash
$ ./arg-test first second w00t "multi word arg"  
The arguments are:  
first  
second  
w00t  
multi word arg  
The program name is:  
arg-test  
```

### Random Numbers

```haskell
-- Random number generators of different types
ghci> random (mkStdGen 949488) :: (Float, StdGen)
(0.8938442,1597344447 1655838864)
ghci> random (mkStdGen 949488) :: (Bool, StdGen)
(False,1485632275 40692)
ghci> random (mkStdGen 949488) :: (Integer, StdGen)
(1691547873,1597344447 1655838864)

-- 3 coin flips
threeCoins :: StdGen -> (Bool, Bool, Bool)
threeCoins gen =
    let (firstCoin, newGen) = random gen
        (secondCoin, newGen') = random newGen
        (thirdCoin, newGen'') = random newGen'
    in  (firstCoin, secondCoin, thirdCoin)
    
-- infinite sequence of random values
ghci> take 5 $ randoms (mkStdGen 11) :: [Int]
[-1807975507,545074951,-1015194702,-1622477312,-502893664]
ghci> take 5 $ randoms (mkStdGen 11) :: [Bool]
[True,True,True,True,False]
ghci> take 5 $ randoms (mkStdGen 11) :: [Float]
[7.904789e-2,0.62691015,0.26363158,0.12223756,0.38291094]

-- Random values in a range
ghci> randomR (1,6) (mkStdGen 359353)
(6,1494289578 40692)
ghci> randomR (1,6) (mkStdGen 35935335)
(3,1250031057 40692)

-- Stream of random values in a range
ghci> take 10 $ randomRs ('a','z') (mkStdGen 3) :: [Char]  
"ndkxbvmomg"
```

Using getStdGen :: IO StdGen from System.Random

```haskell
import System.Random

main = do
    gen <- getStdGen
    putStr $ take 20 (randomRs ('a','z') gen)
```

Use `newStdGen` to get new random number generator

```haskell
import System.Random

main = do
    gen <- getStdGen
    putStrLn $ take 20 (randomRs ('a','z') gen)
    gen' <- newStdGen
    putStr $ take 20 (randomRs ('a','z') gen')
```

### Bytestrings

Better performance for programs that read a lot of data into strings by reading data into chunks, and processing them chunks at a time, reducing IO latency.

```haskell
-- 64K bytes read in per time, rest are lazily loaded
import qualified Data.ByteString.Lazy as B
-- Strict bytestrings
import qualified Data.ByteString as S
```

Example copying files using Bytestrings rather than Strings

```haskell
import System.Environment
import qualified Data.ByteString.Lazy as B

main = do
    (fileName1:fileName2:_) <- getArgs
    copyFile fileName1 fileName2

copyFile :: FilePath -> FilePath -> IO ()
copyFile source dest = do
    contents <- B.readFile source
    B.writeFile dest contents
```

### Exceptions

"Pure code can throw exceptions, but it they can only be caught in the I/O part of our code (when we're inside a do block that goes into main). That's because you don't know when (or if) anything will be evaluated in pure code, because it is lazy and doesn't have a well-defined order of execution, whereas I/O code does." (from Learnyouahaskell)

Don't use exception in the pure part of the code, just use them in I/O only.

```haskell
import System.Environment
import System.IO
import System.IO.Error

main = toTry `catch` handler
   
toTry :: IO ()
toTry = do (fileName:_) <- getArgs
           contents <- readFile fileName
           putStrLn $ "The file has " ++ show (length (lines contents)) ++ " lines!"

handler :: IOError -> IO ()
handler e
    | isDoesNotExistError e =
        case ioeGetFileName e of Just path -> putStrLn $ "Whoops! File does not exist at: " ++ path
                                 Nothing -> putStrLn "Whoops! File does not exist at unknown location!"
    | otherwise = ioError e
```

"Using case expressions is commonly used when you want to pattern match against something without bringing in a new function" (from Learnyouahaskell)

Predicates that act on `IOError`

- `isAlreadyExistsError`
- `isDoesNotExistError`
- `isAlreadyInUseError`
- `isFullError`
- `isEOFError`
- `isIllegalOperation`
- `isPermissionError`
- `isUserError`



### *Monad* typeclass

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

