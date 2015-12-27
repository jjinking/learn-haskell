
-- fp-hello-world-n-times
hello_worlds n = sequence_ [putStrLn "Hello World" | x <- [1..n]]

-- fp-list-replication
listRepl n arr = [x | x <- arr, _ <- [1..n]]

-- fp-filter-array
filterArr n arr = [x | x <- arr, x < n]

-- fp-filter-positions-in-a-list
filterOdd lst = [x | (i, x) <- zip [0..] lst, mod i 2 == 1]

-- fp-array-of-n-elements
zeros' n = [0 | _ <- [1..n]]

-- fp-reverse-a-list
rev l = if null l then [] else last l:rev (init l)
