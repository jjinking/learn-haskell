module Test where


doubleMe :: Int -> Int
doubleMe x = x + x

doubleUs :: Int -> Int -> Int
doubleUs x y = doubleMe x + doubleMe y

doubleSmallNumber :: Int -> Int
doubleSmallNumber x = if x > 100
                        then x
                        else x*2

doubleSmallNumber' :: Int -> Int
doubleSmallNumber' x = (if x > 100 then x else x * 2) + 1

-- Pattern matching
threeOrFive :: (Integral a) => a -> String
threeOrFive 3 = "THREE"
threeOrFive 5 = "FIVE"
threeOrFive x = show (fromIntegral x :: Integer) ++ " IS NOT THREE OR FIVE!"

-- Recursion
factorial :: (Integral a) => a -> a
factorial 0 = 1
factorial n = n * factorial (n - 1)

-- Adding two vectors in R2
addVectors :: (Num a) => (a, a) -> (a, a) -> (a, a)
addVectors a b = ((+) (fst a) (fst b), (+) (snd a) (snd b))

-- Better way
addVectors2 :: (Num a) => (a, a) -> (a, a) -> (a, a)
addVectors2 (x1, y1) (x2, y2) = (x1 + x2, y1 + y2)

-- Use underscore to throw away rest
tell :: (Show a) => [a] -> String
tell [] = "The list is empty"
tell [x] = "The list has one element: " ++ show x
tell [x,y] = "The list has two elements: " ++ show x ++ " and " ++ show y
tell (x:y:_) = "This list has 3 or more elements, with first two elements "++ show x ++ " and " ++ show y

-- Match entire input to use inside function
matchall :: String -> String
matchall "" = "Empty string!"
matchall all@(x:xs) = "The first letter of " ++ all ++ " is " ++ [x]

-- Guards
ageTell :: (RealFloat a) => a -> String
ageTell age
  | age < 18.0 = "You're underage!"
  | age < 21.0 = "You're a young adult!"
  | age < 30.0 = "You're in your twenties!"
  | otherwise  = "You're getting older and wiser!"


replicate' :: (Num i, Ord i) => i -> a -> [a]
replicate' n x
  | n <= 0    = []
  | otherwise = x:replicate' (n-1) x




