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


fooBars xs = [if (x `mod` 3) == 0 then "FOO" else "BAR!" | x <- xs, odd x]
