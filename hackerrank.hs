module HackerRank where

--------------------------------------------------------------------------------
-- Introduction --

-- fp-hello-world-n-times
helloWorlds :: Int -> IO()
helloWorlds n = sequence_ [putStrLn "Hello World" | _ <- [1..n]]

-- fp-list-replication
listRepl :: Int -> [a] -> [a]
listRepl n arr = [x | x <- arr, _ <- [1..n]]

-- fp-filter-array
filterArr :: Int -> [Int] -> [Int]
filterArr n arr = [x | x <- arr, x < n]

-- fp-filter-positions-in-a-list
filterOdd :: [Int] -> [Int]
filterOdd lst = [x | (i, x) <- zip [0..] lst, mod i 2 == 1]

-- fp-array-of-n-elements
zeros' :: Int -> [Int]
zeros' n = [0 | _ <- [1..n]]

-- fp-reverse-a-list
rev :: [a] -> [a]
rev l = if null l then [] else last l:rev (init l)

-- fp-update-list
absVals :: [Int] -> [Int]
absVals arr = [abs x | x <- arr]

-- eval-ex
factorial :: (Integral a) => a -> a
factorial 0 = 1
factorial n = n * factorial (n - 1)
exp' :: Double -> Double
exp' x = sum (take 10 [((x**(fromIntegral i))/(fromIntegral (factorial i))) | i <- [0..]])

-- area-under-curves-and-volume-of-revolving-a-curv
solveAreaVol :: Int -> Int -> [Int] -> [Int] -> [Double]
solveAreaVol l r a b = [sum (fst areaVol), sum (snd areaVol)] where areaVol = unzip [(ai, ai * f_xi * pi) | f_xi <- evalPolys l r a b, let ai = f_xi * 0.001]
evalPolys :: Int -> Int -> [Int] -> [Int] -> [Double]
evalPolys l r a b = [evalPoly a b xi | xi <- intervalRange l r]
evalPoly :: [Int] -> [Int] -> Double -> Double
evalPoly a b x = sum [(fromIntegral ai) * (x^^bi) | (ai, bi) <- zip a b]
intervalRange :: Int -> Int -> [Double]
intervalRange l r = map (/1000) [fromIntegral x | x <- [l*1000..r*1000]]

-- fp-sum-of-odd-elements
sumOdd :: [Int] -> Int
sumOdd arr = sum [x | x <- arr, odd x]

-- fp-list-length
len' :: [a] -> Int
len' lst = sum [1 | _ <- lst]

--------------------------------------------------------------------------------
-- Recursion

-- functional-programming-warmups-in-recursion---gcd
gcd' :: Integral a => a -> a -> a
gcd' n m
  | n == m = m
  | n < m = gcd' m n
  | otherwise = gcd' (n-m) m

-- functional-programming-warmups-in-recursion---fibonacci-numbers
fib' :: Int -> Int
fib' 0 = 0
fib' 1 = 0
fib' 2 = 1
fib' n = fib' (n-1) + fib' (n-2)
